"""ROS policy that drives the aic engine with a fine-tuned X-VLA checkpoint
served by `aic_xvla.serve` (a separate process running in the X-VLA conda
env). The ROS process here only deals with ROS messages + HTTP, so it
runs cleanly inside the aic pixi env without X-VLA's torch dependency.

Run (after `aic_xvla.serve` is up and `/entrypoint.sh` started the engine):

    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=aic_xvla.ros.RunXVLA \\
        -p policy_config_file:=/path/to/runxvla.yaml   # optional

Configuration via env vars:
    AIC_XVLA_SERVER_URL        default http://127.0.0.1:8010
    AIC_XVLA_TIMEOUT_S         default 30.0
    AIC_XVLA_REPLAN            default 1     (every N executed actions before requerying)
    AIC_XVLA_TASK_TIMEOUT_S    default 60.0
    AIC_XVLA_CONTROL_PERIOD_S  default 0.25

Debug aids (optional, off by default):
    AIC_XVLA_DEBUG_IMAGE_DIR   if set, the first observed frame from each cam
                               is written here as PNG so you can visually
                               compare against the training JPEGs.
    AIC_XVLA_BGR_INPUT=1       treat ROS image bytes as BGR instead of RGB
                               (toggle to test the channel-order hypothesis
                               without recompiling).
"""

from __future__ import annotations

import base64
import io
import os
import time

import cv2
import numpy as np
import requests
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)

DEFAULT_INSTRUCTION = "insert the SFP cable into the port"


def _state_from_observation(obs: Observation) -> np.ndarray:
    """Build the same 26D state vector the aic data parquets store."""
    tcp_pose = obs.controller_state.tcp_pose
    tcp_vel = obs.controller_state.tcp_velocity
    return np.array(
        [
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
            tcp_vel.linear.x,
            tcp_vel.linear.y,
            tcp_vel.linear.z,
            tcp_vel.angular.x,
            tcp_vel.angular.y,
            tcp_vel.angular.z,
            *obs.controller_state.tcp_error,  # 6
            *obs.joint_states.position[:7],  # 7
        ],
        dtype=np.float64,
    )


def _ros_image_to_b64(img_msg, bgr_input: bool = False) -> str:
    arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    # The data parquet's JPEGs are RGB-encoded. cv2 wants BGR for imencode,
    # so convert from whatever the live source actually is.
    bgr = arr if bgr_input else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class RunXVLA(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.server_url = os.environ.get("AIC_XVLA_SERVER_URL", "http://127.0.0.1:8010")
        self.timeout_s = float(os.environ.get("AIC_XVLA_TIMEOUT_S", "30"))
        self.replan_every = int(os.environ.get("AIC_XVLA_REPLAN", "1"))
        self.task_timeout_s = float(os.environ.get("AIC_XVLA_TASK_TIMEOUT_S", "60"))
        self.control_period_s = float(
            os.environ.get("AIC_XVLA_CONTROL_PERIOD_S", "0.25")
        )
        self.debug_image_dir = os.environ.get("AIC_XVLA_DEBUG_IMAGE_DIR", "") or None
        self.bgr_input = os.environ.get("AIC_XVLA_BGR_INPUT", "0") == "1"
        self._dumped_images = False
        self.get_logger().info(
            f"RunXVLA server={self.server_url} replan_every={self.replan_every} "
            f"timeout_s={self.timeout_s} control_period_s={self.control_period_s} "
            f"debug_image_dir={self.debug_image_dir} bgr_input={self.bgr_input}"
        )
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        for _ in range(10):
            try:
                r = requests.get(f"{self.server_url}/healthz", timeout=2.0)
                if r.ok:
                    self.get_logger().info(f"server healthy: {r.json()}")
                    return
            except Exception as e:
                self.get_logger().info(f"waiting for server ({e})")
                time.sleep(1.0)
        self.get_logger().warn(
            f"server at {self.server_url} not responding; continuing anyway"
        )

    def _maybe_dump_first_frame(self, obs: Observation) -> None:
        if not self.debug_image_dir or self._dumped_images:
            return
        try:
            os.makedirs(self.debug_image_dir, exist_ok=True)
            for cam, msg in [
                ("left", obs.left_image),
                ("center", obs.center_image),
                ("right", obs.right_image),
            ]:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                # Write two versions so you can tell at a glance which channel order is right.
                cv2.imwrite(
                    os.path.join(self.debug_image_dir, f"first_{cam}_assume_rgb.png"),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(self.debug_image_dir, f"first_{cam}_assume_bgr.png"),
                    arr,
                )
            self._dumped_images = True
            self.get_logger().info(f"dumped first frames to {self.debug_image_dir}")
        except Exception as ex:
            self.get_logger().warn(f"image dump failed: {ex}")

    def _request_actions(self, obs: Observation, instruction: str) -> np.ndarray:
        state = _state_from_observation(obs)
        payload = {
            "state": state.tolist(),
            "images": [
                _ros_image_to_b64(obs.left_image, self.bgr_input),
                _ros_image_to_b64(obs.center_image, self.bgr_input),
                _ros_image_to_b64(obs.right_image, self.bgr_input),
            ],
            "instruction": instruction,
            "steps": 10,
        }
        r = requests.post(
            f"{self.server_url}/act", json=payload, timeout=self.timeout_s
        )
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"server error: {body['error']}")
        return np.asarray(body["actions"], dtype=np.float32)  # [T, 7]

    def _send_pose(self, move_robot: MoveRobotCallback, action: np.ndarray) -> None:
        pose = Pose(
            position=Point(x=float(action[0]), y=float(action[1]), z=float(action[2])),
            orientation=Quaternion(
                x=float(action[3]),
                y=float(action[4]),
                z=float(action[5]),
                w=float(action[6]),
            ),
        )
        self.set_pose_target(move_robot=move_robot, pose=pose)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.get_logger().info(f"RunXVLA.insert_cable() enter. Task: {task}")
        instruction = getattr(task, "instruction", "") or DEFAULT_INSTRUCTION
        self.get_logger().info(f"using instruction: {instruction!r}")
        send_feedback("RunXVLA: starting")

        start = time.time()
        chunk: np.ndarray | None = None
        chunk_idx = 0
        step = 0

        while time.time() - start < self.task_timeout_s:
            loop_start = time.time()
            obs = get_observation()
            if obs is None:
                self.sleep_for(self.control_period_s)
                continue

            self._maybe_dump_first_frame(obs)
            live_pos = (
                obs.controller_state.tcp_pose.position.x,
                obs.controller_state.tcp_pose.position.y,
                obs.controller_state.tcp_pose.position.z,
            )

            need_replan = chunk is None or chunk_idx >= min(
                self.replan_every, chunk.shape[0]
            )
            if need_replan:
                try:
                    t_inf = time.time()
                    chunk = self._request_actions(obs, instruction)
                    chunk_idx = 0
                    self.get_logger().info(
                        f"chunk shape={tuple(chunk.shape)} inf_dt={time.time()-t_inf:.2f}s"
                    )
                except Exception as ex:
                    self.get_logger().error(f"inference failed: {ex}")
                    send_feedback(f"inference error: {ex}")
                    self.sleep_for(self.control_period_s)
                    continue

            action = chunk[chunk_idx]
            chunk_idx += 1
            self._send_pose(move_robot, action)
            self.get_logger().info(
                f"step={step} live_pos=[{live_pos[0]:+.4f},{live_pos[1]:+.4f},{live_pos[2]:+.4f}] "
                f"target_pos=[{action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}] "
                f"Δz={action[2]-live_pos[2]:+.4f}"
            )
            send_feedback(f"step pos={action[:3].round(3).tolist()}")
            step += 1

            elapsed = time.time() - loop_start
            self.sleep_for(max(0.0, self.control_period_s - elapsed))

        self.get_logger().info("RunXVLA.insert_cable() exiting (timeout)")
        return True
