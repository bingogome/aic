#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

# pi05 env only: LD_PRELOAD conda's libtiff/libjpeg so ROS (which else would
# load the system /lib/x86_64-linux-gnu/libtiff.so.6 — built against libjpeg8
# expecting symbol `jpeg12_write_raw_data` that conda-forge libjpeg-turbo
# does not export) never pulls them in first. Observed symptom in default
# eval: `import cv2` in the policy action thread crashes with
# `ImportError: libtiff.so.6: undefined symbol: jpeg12_write_raw_data`.
# Guarded on CONDA_PREFIX so the default env is not affected.
case "${CONDA_PREFIX:-}" in
  */envs/pi05)
    _pi05_libtiff="${CONDA_PREFIX}/lib/libtiff.so.6"
    _pi05_libjpeg="${CONDA_PREFIX}/lib/libjpeg.so.8"
    if [[ -f "${_pi05_libtiff}" && -f "${_pi05_libjpeg}" ]]; then
      export LD_PRELOAD="${_pi05_libjpeg}:${_pi05_libtiff}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
    unset _pi05_libtiff _pi05_libjpeg
    ;;
esac
