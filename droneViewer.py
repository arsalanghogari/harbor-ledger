

# ABOUT THIS SCRIPT
# This is a simple simulation script that simply renders the robot/drone in an empty simulation environment.
# This script is fast/efficient enough to run in real-time on CPU (and does NOT lag or crash your computer).

# HOW TO RUN THIS SCRIPT
# PRE-REQUISITES:
# - Running this script requires Python and the MuJoCo library installed via pip:
#     pip install mujoco
# Run:
#   python droneViewer.py
#
# To switch models, change the MODEL_PATH global variable below.

import time
from pathlib import Path

import mujoco
import mujoco.viewer

# ==============================================================================
# Model Selection (Global)
# ==============================================================================
# Drone:
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "drone_assets" / "skydio_x2" / "scene.xml"

# Quadruped Dog:
# MODEL_PATH = "mujoco_menagerie/unitree_go2/scene.xml"

# Humanoids:
# MODEL_PATH = "mujoco_menagerie/unitree_g1/scene.xml"
# MODEL_PATH = "mujoco_menagerie/unitree_h1/scene.xml"

import numpy as np

def get_drone_state(model, data, body_name="x2"):
    """
    Read the drone's 6-DOF state.

    Position and quaternion are in world coordinates.
    Linear velocity is in world coordinates.
    Angular velocity is in the body frame.
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found")

    # World-frame position
    position = data.xpos[body_id].copy()

    # World-frame orientation quaternion (w, x, y, z)
    quaternion = data.xquat[body_id].copy()

    # Spatial velocity: [angular, linear]
    spatial_velocity = np.zeros(6)

    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        spatial_velocity,
        0  # World-frame representation
    )

    angular_velocity = spatial_velocity[:3].copy()
    linear_velocity = spatial_velocity[3:].copy()

    return {
        "time": float(data.time),
        "position": position,
        "quaternion": quaternion,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity
    }

def main():
    # Load model and data
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print(f"Successfully loaded model from {MODEL_PATH}")
    print("Stepping simulation in a loop...")

    # Step simulation loop (with passive viewer if available, fallback to headless loop)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                print( get_drone_state (model, data) )
                mujoco.mj_step(model, data)
                viewer.sync()

                # Sync rate to real time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        print(f"Viewer launch not available or running headless: {e}")
        print("Running 1000 steps headlessly...")
        for i in range(1000):
            mujoco.mj_step(model, data)
        print("Completed 1000 simulation steps successfully.")


if __name__ == "__main__":
    main()
