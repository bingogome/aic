from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger

from ch_milestones.environment.services import call


class ResetClient:
    def __init__(self, node):
        self.node = node
        self.parameter_client = node.create_client(
            SetParameters, "/ch_milestone_environment_resetter/set_parameters"
        )
        self.reset_client = node.create_client(Trigger, "/ch_milestones/reset_episode")
        self.clear_client = node.create_client(
            Trigger, "/ch_milestones/clear_environment"
        )

    def reset(self, timeout_sec, task_index=None):
        if task_index is not None:
            self.select_task_index(task_index, timeout_sec)
        response = call(
            self.node,
            self.reset_client,
            Trigger.Request(),
            timeout_sec,
            "reset episode",
        )
        if not response.success:
            raise RuntimeError(response.message)

    def select_task_index(self, task_index: int, timeout_sec: float):
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="selected_task_index",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=int(task_index),
                ),
            )
        ]
        response = call(
            self.node,
            self.parameter_client,
            request,
            timeout_sec,
            "select reset task index",
        )
        for result in response.results:
            if not result.successful:
                raise RuntimeError(result.reason)

    def clear(self, timeout_sec):
        response = call(
            self.node,
            self.clear_client,
            Trigger.Request(),
            timeout_sec,
            "clear environment",
        )
        if not response.success:
            raise RuntimeError(response.message)
