"""
ABOUT THIS SCRIPT
This simulation script renders the Skydio X2 drone in MuJoCo with a high-speed,
responsive racing flight controller that supports true physical key presses & continuous holds
(via Windows GetAsyncKeyState API and GLFW).

Key Features:
- True Press & Hold Detection: Zero OS repeat delay, zero stuttering. Holding keys
  accelerates smoothly; releasing immediately engages auto-braking / auto-leveling.
- Racing Drone Agility & Speed:
  - Fast Racing Tilt (up to 40°-48° tilt) for aggressive forward acceleration (15+ m/s).
  - High-power vertical climb and dive rates (up to 4.5-6.0 m/s).
  - Snappy yaw rates (200°+/sec) for carving tight corners.
  - Turbo Boost / Sprint: Hold SHIFT while flying for nitro racing speeds!
  - Switchable Flight Modes: Toggle between SPORT/RACING mode and CRUISE mode (Key: 'M').
- Dual Rendering:
  1) Interactive 3D MuJoCo scene viewer.
  2) Compact, low-latency OpenCV window displaying real-time RGB & Depth sensor readings with flight telemetry.
- Dynamic Scene Support: Seamlessly switch between empty environment (--scene none),
  SceneSmith Office mesh, or drone racing gates.

HOW TO RUN THIS SCRIPT
PRE-REQUISITES:
- Python with MuJoCo:
    pip install mujoco
- Optional RGB-D window support:
    pip install opencv-python

RUNNING:
- Run in empty space (default testing environment):
    python droneViewer2.py --scene none
- Run inside SceneSmith Office indoor mesh:
    python droneViewer2.py --scene Scenes/scenesmith_office/scene.xml
- Run with Drone Racing Gates:
    python droneViewer2.py --scene Scenes/gates.xml
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ==============================================================================
# Model & Environment Scene Selection (Global Defaults)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / "drone_assets" / "skydio_x2" / "scene.xml")
DRONE_BODY_XML = str(SCRIPT_DIR / "drone_assets" / "skydio_x2" / "x2.xml")

# Dynamic Environment Scene Path:
# Set to "none" for empty scene, or path to an environment scene XML
ENV_SCENE_PATH = "none"
# ENV_SCENE_PATH = r"Scenes\restaurant_3x\scene.xml"

# Camera sensor settings
CAMERA_NAME = "depth_camera"  # Front-facing depth camera (falls back to "fpv" if not found)
CAM_WIDTH = 160               # Small resolution for maximum computational efficiency
CAM_HEIGHT = 120
RENDER_FPS = 30               # Target camera sensor update rate (Hz)
WINDOW_NAME = "Skydio X2 - Front RGB-D Camera (Monocular Depth)"

# Depth visualization range in meters
DEPTH_MIN_METERS = 0.2
DEPTH_MAX_METERS = 8.0

# Spawn positions [X, Y, Z] (meters)
DEFAULT_SPAWN_POS = [0.0, 0.0, 0.5]
OFFICE_SPAWN_POS = [1.8, 1.2, 1.0]


# ==============================================================================
# Continuous Physical Keyboard Press & Hold Detector
# ==============================================================================
class ContinuousKeyboard:
    """
    Direct asynchronous hardware keyboard poller.
    Accurately detects continuous key holds with zero OS repeat delay.
    Uses Win32 GetAsyncKeyState on Windows, with GLFW/OpenCV event fallback.
    """
    def __init__(self):
        self.is_windows = (os.name == 'nt')
        if self.is_windows:
            import ctypes
            self.user32 = ctypes.windll.user32
            # Virtual-Key codes
            self.VK_MAP = {
                'W': 0x57, 'S': 0x53, 'A': 0x41, 'D': 0x44,
                'I': 0x49, 'K': 0x4B, 'J': 0x4A, 'L': 0x4C,
                'H': 0x48, 'R': 0x52, 'M': 0x4D, 'C': 0x43, 'B': 0x42,
                'SPACE': 0x20, 'SHIFT': 0x10, 'ESC': 0x1B,
                'UP': 0x26, 'DOWN': 0x28, 'LEFT': 0x25, 'RIGHT': 0x27,
                'Q': 0x51, 'E': 0x45
            }
        self.active_keys = set()
        self.key_last_seen = {}

    def is_down(self, key_name: str) -> bool:
        """Returns True if the key is currently physically held down."""
        if self.is_windows and key_name in self.VK_MAP:
            vk = self.VK_MAP[key_name]
            return bool(self.user32.GetAsyncKeyState(vk) & 0x8000)
        # Fallback for event-based key hold tracking
        return (time.time() - self.key_last_seen.get(key_name, 0.0)) < 0.15

    def register_event_key(self, key_name: str):
        self.key_last_seen[key_name] = time.time()

    def any_down(self, *key_names) -> bool:
        return any(self.is_down(k) for k in key_names)


# ==============================================================================
# Realistic Smooth Closed-Loop Physics Drone Flight Controller
# ==============================================================================
class RacingFlightController:
    """
    Smooth, realistic closed-loop physics quadrotor flight controller.
    Features:
      - Realistic multi-rotor physics: Computes individual motor thrusts (f1, f2, f3, f4)
        acting through MuJoCo's physics engine with true aerodynamic lift, roll/pitch moments,
        and yaw counter-torque.
      - Smooth aerodynamic banking & pitching: The drone naturally pitches forward to accelerate,
        banks smoothly into turns, and climbs/descends with realistic inertia.
      - Altitude PID with Dynamic Tilt Compensation: Automatically increases total thrust by
        1/cos(tilt) during banking/pitching maneuvers to maintain altitude without sag.
      - Critically Damped Attitude PD: Returns to level orientation smoothly without overshoot,
        ringing, or wobble upon key release.
      - Active Auto-Braking: Decelerates smoothly when keys are released and locks into a stable hover.
      - Proportional Motor Desaturation: Preserves attitude control authority during high-thrust maneuvers.
    """
    def __init__(self, mass=1.325, lx=0.14, ly=0.18, c_yaw=0.0201, max_thrust_per_motor=13.0):
        self.mass = mass
        self.g = 9.81
        self.lx = lx
        self.ly = ly
        self.c_yaw = c_yaw
        self.max_thrust = max_thrust_per_motor
        self.hover_thrust = (self.mass * self.g) / 4.0

        # State targets
        self.target_z = 0.5
        self.target_yaw = 0.0
        self.cmd_pitch = 0.0
        self.cmd_roll = 0.0
        self.flight_mode_name = "SPORT"
        self.action_label = "HOVER-LOCKED"
        self.camera_mode_name = "FREE"
        self._active_action_str = ""

        # Per-axis active input tracking
        self.pitch_active = False
        self.roll_active = False
        self.climb_active = False
        self.yaw_active = False
        self.prev_climb_active = False
        self.prev_yaw_active = False

        # Auto-braking configuration
        self.autobrake_enabled = True
        self.Kv_brake = 0.30          # Counter-tilt per m/s horizontal speed
        self.max_brake_tilt = 0.52    # Max counter-tilt reverse flare (~30°)
        self.max_total_tilt = 0.78    # Max compound tilt (~45°)
        self.stop_threshold = 0.15    # Speed threshold (m/s) below which drone settles to level
        self.x_lock = None
        self.y_lock = None

        # Speed profiles
        self.profiles = {
            "SPORT": {
                "max_tilt": 0.65,       # ~37° tilt (racing speed ~15 m/s)
                "turbo_tilt": 0.80,     # ~46° tilt (sprint speed ~20 m/s)
                "climb_speed": 3.2,     # m/s
                "turbo_climb": 5.0,     # m/s
                "yaw_speed": 3.0,       # rad/s (~172°/s)
                "ramp_rate": 8.0,       # rad/s^2 smooth stick response
            },
            "CRUISE": {
                "max_tilt": 0.35,       # ~20° tilt (smooth cinematic speed)
                "turbo_tilt": 0.48,     # ~27° tilt
                "climb_speed": 1.8,     # m/s
                "turbo_climb": 2.8,     # m/s
                "yaw_speed": 1.8,       # rad/s (~103°/s)
                "ramp_rate": 5.0,
            }
        }
        self.current_profile = self.profiles["SPORT"]

        # Tuned High-Performance Critically Damped PID Gains
        self.Kp_z = 35.0
        self.Kd_z = 12.0

        self.Kp_roll = 4.2
        self.Kd_roll = 0.45

        self.Kp_pitch = 4.2
        self.Kd_pitch = 0.45

        self.Kp_yaw = 2.0
        self.Kd_yaw = 0.30

    def reset(self, initial_pos, initial_yaw=0.0):
        """Reset targets to current position and heading."""
        self.target_z = float(initial_pos[2])
        self.target_yaw = float(initial_yaw)
        self.cmd_pitch = 0.0
        self.cmd_roll = 0.0
        self.pitch_active = False
        self.roll_active = False
        self.climb_active = False
        self.yaw_active = False
        self.x_lock = float(initial_pos[0])
        self.y_lock = float(initial_pos[1])
        self.action_label = "HOVER-LOCKED"
        self._active_action_str = ""

    def toggle_mode(self):
        """Toggle between SPORT (racing) and CRUISE mode."""
        if self.flight_mode_name == "SPORT":
            self.flight_mode_name = "CRUISE"
        else:
            self.flight_mode_name = "SPORT"
        self.current_profile = self.profiles[self.flight_mode_name]
        print(f"\n>>> Flight Profile Switched to: {self.flight_mode_name} MODE <<<")

    def toggle_autobrake(self):
        """Toggle the Auto-Braking system ON or OFF."""
        self.autobrake_enabled = not self.autobrake_enabled
        status_str = "ENABLED (Active Stop & Hover Lock)" if self.autobrake_enabled else "DISABLED (Manual Coasting)"
        print(f"\n>>> Auto-Braking System: {status_str} <<<")

    @staticmethod
    def quat_to_euler(w, x, y, z):
        """Converts quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians."""
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def update_from_keyboard(self, kb, dt: float):
        """
        Continuously samples hardware key hold states and updates flight targets.
        Enables smooth physical response with per-axis auto-brake activation.
        """
        is_turbo = kb.is_down('SHIFT')
        prof = self.current_profile

        climb_rate = prof["turbo_climb"] if is_turbo else prof["climb_speed"]
        max_tilt = prof["turbo_tilt"] if is_turbo else prof["max_tilt"]
        yaw_rate = prof["yaw_speed"]
        ramp = prof["ramp_rate"]

        actions = []

        # 1. Altitude (Climb / Descend)
        self.climb_active = False
        if kb.any_down('W', 'SPACE'):
            self.target_z += climb_rate * dt
            self.climb_active = True
            actions.append("CLIMB")
        elif kb.any_down('S', 'C'):
            self.target_z = max(0.12, self.target_z - climb_rate * dt)
            self.climb_active = True
            actions.append("DESCEND")

        # 2. Yaw (Turn Left / Right)
        self.yaw_active = False
        if kb.any_down('A', 'Q'):
            self.target_yaw += yaw_rate * dt
            self.yaw_active = True
            actions.append("YAW-L")
        elif kb.any_down('D', 'E'):
            self.target_yaw -= yaw_rate * dt
            self.yaw_active = True
            actions.append("YAW-R")

        # 3. Pitch Axis (Forward / Backward)
        self.pitch_active = False
        desired_pitch = 0.0
        if kb.any_down('I', 'UP'):
            desired_pitch = max_tilt
            self.pitch_active = True
            self.x_lock = None
            actions.append("FWD" if not is_turbo else "TURBO-FWD")
        elif kb.any_down('K', 'DOWN'):
            desired_pitch = -max_tilt
            self.pitch_active = True
            self.x_lock = None
            actions.append("BACK")

        # 4. Roll Axis (Strafe Left / Right)
        self.roll_active = False
        desired_roll = 0.0
        if kb.any_down('J', 'LEFT'):
            desired_roll = -max_tilt
            self.roll_active = True
            self.y_lock = None
            actions.append("STRAFE-L")
        elif kb.any_down('L', 'RIGHT'):
            desired_roll = max_tilt
            self.roll_active = True
            self.y_lock = None
            actions.append("STRAFE-R")

        # Smooth exponential stick ramping when driven; instant release when stick released
        alpha = min(1.0, ramp * dt)
        if self.pitch_active:
            self.cmd_pitch = (1.0 - alpha) * self.cmd_pitch + alpha * desired_pitch
        else:
            self.cmd_pitch = 0.0  # Immediately free pitch axis for active auto-brake

        if self.roll_active:
            self.cmd_roll = (1.0 - alpha) * self.cmd_roll + alpha * desired_roll
        else:
            self.cmd_roll = 0.0   # Immediately free roll axis for active auto-brake

        # Emergency Hover Lock key (H)
        if kb.is_down('H'):
            self.pitch_active = False
            self.roll_active = False
            self.cmd_pitch = 0.0
            self.cmd_roll = 0.0
            self.x_lock = None
            self.y_lock = None
            actions = ["HOVER-LOCK"]

        self._active_action_str = " + ".join(actions) if actions else ""

    def compute_control(self, data, dt: float) -> np.ndarray:
        """
        Computes 4 motor thrusts (f1, f2, f3, f4) in Newtons.
        Combines Altitude PID, Attitude PD, and Per-Axis Active Auto-Braking.
        """
        w, x, y, z = data.qpos[3:7]
        roll, pitch, yaw = self.quat_to_euler(w, x, y, z)
        p, q, r = data.qvel[3:6]
        vx, vy, vz = data.qvel[0:3]

        # Convert horizontal velocities to drone body frame
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        vx_body = cos_y * vx + sin_y * vy
        vy_body = -sin_y * vx + cos_y * vy

        # Snap target altitude when climb/descend keys are released
        if self.prev_climb_active and not self.climb_active:
            self.target_z = float(data.qpos[2])
        self.prev_climb_active = self.climb_active

        # Snap target yaw when yaw keys are released
        if self.prev_yaw_active and not self.yaw_active:
            self.target_yaw = float(yaw)
        self.prev_yaw_active = self.yaw_active

        # ----------------------------------------------------------------------
        # Per-Axis Auto-Braking Logic:
        # If key is NOT pressed on an axis, apply active counter-tilt to arrest
        # velocity along that specific axis until it halts and locks.
        # ----------------------------------------------------------------------
        brake_tags = []

        # --- Pitch Axis (Forward / Backward) ---
        if self.pitch_active or not self.autobrake_enabled:
            raw_pitch = self.cmd_pitch
            self.x_lock = None
        else:
            if abs(vx_body) > self.stop_threshold:
                raw_pitch = np.clip(-self.Kv_brake * vx_body, -self.max_brake_tilt, self.max_brake_tilt)
                brake_tags.append("BRAKE-X")
            else:
                raw_pitch = 0.0

        # --- Roll Axis (Strafe Left / Right) ---
        if self.roll_active or not self.autobrake_enabled:
            raw_roll = self.cmd_roll
            self.y_lock = None
        else:
            if abs(vy_body) > self.stop_threshold:
                raw_roll = np.clip(self.Kv_brake * vy_body, -self.max_brake_tilt, self.max_brake_tilt)
                brake_tags.append("BRAKE-Y")
            else:
                raw_roll = 0.0

        # Vector tilt magnitude limit to prevent motor saturation during compound maneuvers
        tilt_mag = np.hypot(raw_pitch, raw_roll)
        if tilt_mag > self.max_total_tilt:
            scale = self.max_total_tilt / tilt_mag
            target_pitch = raw_pitch * scale
            target_roll = raw_roll * scale
        else:
            target_pitch = raw_pitch
            target_roll = raw_roll

        # Set human-readable HUD action label
        if self._active_action_str:
            self.action_label = self._active_action_str
        elif brake_tags:
            self.action_label = " + ".join(brake_tags)
        elif self.autobrake_enabled:
            self.action_label = "HOVER-LOCKED"
        else:
            self.action_label = "COASTING"

        # ----------------------------------------------------------------------
        # 1. Altitude PID Controller with dynamic tilt factor compensation
        # ----------------------------------------------------------------------
        e_z = self.target_z - data.qpos[2]
        tilt_factor = max(0.38, np.cos(roll) * np.cos(pitch))
        thrust_total = (self.mass * self.g + self.Kp_z * e_z - self.Kd_z * vz) / tilt_factor
        t0 = np.clip(thrust_total / 4.0, 1.0, 9.5)

        # ----------------------------------------------------------------------
        # 2. Attitude PD Controller
        # ----------------------------------------------------------------------
        e_roll = target_roll - roll
        e_pitch = target_pitch - pitch
        e_yaw = (self.target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]

        tau_x = self.Kp_roll * e_roll - self.Kd_roll * p
        tau_y = self.Kp_pitch * e_pitch - self.Kd_pitch * q
        tau_z = self.Kp_yaw * e_yaw - self.Kd_yaw * r

        # Differential thrust per axis
        df_roll = tau_x / (4.0 * self.ly)
        df_pitch = tau_y / (4.0 * self.lx)
        df_yaw = tau_z / (4.0 * self.c_yaw)

        # Proportional Headroom Desaturation:
        # Scales differential torques if they exceed available motor headroom around t0.
        # Guarantees average thrust remains identically t0 (zero altitude sag)
        # while preserving exact torque direction ratios without clipping.
        headroom = min(t0 - 0.2, self.max_thrust - t0)
        max_df = max(
            abs(-df_roll + df_pitch - df_yaw),
            abs(df_roll + df_pitch + df_yaw),
            abs(df_roll - df_pitch - df_yaw),
            abs(-df_roll - df_pitch + df_yaw)
        )
        if max_df > headroom and max_df > 1e-4:
            scale = max(0.05, headroom / max_df)
            df_roll *= scale
            df_pitch *= scale
            df_yaw *= scale

        # Analytical Motor Mixer for Skydio X2 geometry:
        # Motor 1: rear-right (-x, -y, -c_yaw)
        # Motor 2: rear-left  (-x, +y, +c_yaw)
        # Motor 3: front-left (+x, +y, -c_yaw)
        # Motor 4: front-right(+x, -y, +c_yaw)
        f_arr = np.array([
            t0 - df_roll + df_pitch - df_yaw,
            t0 + df_roll + df_pitch + df_yaw,
            t0 + df_roll - df_pitch - df_yaw,
            t0 - df_roll - df_pitch + df_yaw
        ])

        ctrl = np.clip(f_arr, 0.0, self.max_thrust)
        return ctrl


# Backward-compatibility aliases
SmoothFlightController = RacingFlightController
DirectFlightController = RacingFlightController


# ==============================================================================
# Model Composition & Dynamic Loading
# ==============================================================================
def _compose_mesh_scene(env_scene_path: str, drone_xml_path: str, spawn_pos: list, no_collision: bool = False) -> mujoco.MjModel:
    """Composes a full indoor mesh scene (e.g. SceneSmith Office, Warehouse, Restaurant) with the Skydio drone."""
    env_dir = os.path.dirname(os.path.abspath(env_scene_path)).replace("\\", "/")
    drone_assets_dir = os.path.join(os.path.dirname(os.path.abspath(drone_xml_path)), "assets").replace("\\", "/")

    drone_tree = ET.parse(drone_xml_path)
    drone_root = drone_tree.getroot()

    for tex in drone_root.findall(".//texture"):
        f = tex.get("file")
        if f and not os.path.isabs(f):
            tex.set("file", os.path.join(drone_assets_dir, f).replace("\\", "/"))

    for mesh in drone_root.findall(".//mesh"):
        f = mesh.get("file")
        if f and not os.path.isabs(f):
            mesh.set("file", os.path.join(drone_assets_dir, f).replace("\\", "/"))

    for default_tag in drone_root.findall(".//default"):
        if default_tag.get("class") == "collision":
            for g in default_tag.findall("geom"):
                if no_collision:
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
                else:
                    g.set("contype", "2")
                    g.set("conaffinity", "1")

    env_tree = ET.parse(env_scene_path)
    env_root = env_tree.getroot()

    comp = env_root.find("compiler")
    if comp is not None:
        comp.set("meshdir", os.path.join(env_dir, "meshes").replace("\\", "/"))
        comp.set("texturedir", os.path.join(env_dir, "meshes").replace("\\", "/"))

    wb = env_root.find("worldbody")
    if wb is not None:
        for b in wb.findall(".//body"):
            for j in list(b.findall("joint")):
                b.remove(j)
            for fj in list(b.findall("freejoint")):
                b.remove(fj)

        if no_collision:
            # Strip pure collision-primitive geoms and ensure all visual geoms have zero collision interaction
            for g in list(wb.findall(".//geom")):
                geom_name = g.get("name", "")
                geom_type = g.get("type", "")
                if geom_name.startswith("vis_") or geom_name == "ground_plane" or geom_type == "plane":
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
                else:
                    wb.remove(g)
            print("[*] Real-Time Feasibility Mode: All collisions DISABLED. Using pure textured visual mesh (0 collision geoms).")
        else:
            for g in wb.findall(".//geom"):
                if g.get("type") != "mesh":
                    g.set("contype", "1")
                    g.set("conaffinity", "0")
                else:
                    mesh_name = g.get("mesh", "")
                    if "collision" in mesh_name or g.get("group") == "3" or "wall" in mesh_name or "floor" in mesh_name:
                        g.set("contype", "1")
                        g.set("conaffinity", "0")
                    else:
                        g.set("contype", "0")
                        g.set("conaffinity", "0")

    env_asset = env_root.find("asset")
    drone_asset = drone_root.find("asset")
    if env_asset is not None and drone_asset is not None:
        for child in list(drone_asset):
            env_asset.append(child)

    env_default = env_root.find("default")
    if env_default is None:
        env_default = ET.SubElement(env_root, "default")
    drone_default = drone_root.find("default")
    if drone_default is not None:
        for child in list(drone_default):
            env_default.append(child)

    drone_worldbody = drone_root.find("worldbody")
    if wb is not None and drone_worldbody is not None:
        for child in list(drone_worldbody):
            if child.tag == "body" and child.get("name") == "x2":
                child.set("pos", f"{spawn_pos[0]} {spawn_pos[1]} {spawn_pos[2]}")
            wb.append(child)

    for tag in ["actuator", "sensor", "keyframe"]:
        drone_tag = drone_root.find(tag)
        if drone_tag is not None:
            env_tag = env_root.find(tag)
            if env_tag is None:
                env_tag = ET.SubElement(env_root, tag)
            for child in list(drone_tag):
                if tag == "keyframe" and child.get("name") == "hover":
                    child.set("qpos", f"{spawn_pos[0]} {spawn_pos[1]} {spawn_pos[2]} 1 0 0 0")
                env_tag.append(child)

    # Ensure custom twilight skybox and reflective ground plane are active in the composed environment
    import createDefaultGroundSkyAssets
    texturedir = comp.get("texturedir") if (comp is not None and comp.get("texturedir")) else env_dir
    if not os.path.isabs(texturedir):
        texturedir = os.path.join(env_dir, texturedir)

    rel_sky = os.path.relpath(createDefaultGroundSkyAssets.SKY_TWILIGHT_TEXTURE, texturedir).replace("\\", "/")
    rel_ground = os.path.relpath(createDefaultGroundSkyAssets.GROUND_REFLECTIVE_TEXTURE, texturedir).replace("\\", "/")

    if env_asset is not None:
        sky = env_asset.find(".//texture[@type='skybox']")
        if sky is None:
            sky = ET.SubElement(env_asset, "texture", type="skybox")
        sky.set("file", rel_sky)
        sky.set("gridsize", "3 4")
        sky.set("gridlayout", ".U..LFRB.D..")

        gp_tex = env_asset.find(".//texture[@name='groundplane']")
        if gp_tex is None:
            gp_tex = ET.SubElement(env_asset, "texture", type="2d", name="groundplane")
        gp_tex.set("file", rel_ground)

        gp_mat = env_asset.find(".//material[@name='groundplane']")
        if gp_mat is None:
            gp_mat = ET.SubElement(env_asset, "material", name="groundplane")
        gp_mat.set("texture", "groundplane")
        gp_mat.set("texuniform", "true")
        gp_mat.set("texrepeat", "3 3")
        gp_mat.set("reflectance", "0.42")
        gp_mat.set("specular", "0.65")
        gp_mat.set("shininess", "0.85")

    if wb is not None:
        gp_geom = None
        for g in wb.findall(".//geom"):
            if g.get("name") in ("ground_plane", "floor") or g.get("type") == "plane":
                gp_geom = g
                break
        if gp_geom is None:
            ET.SubElement(wb, "geom", name="ground_plane", size="0 0 0.05", pos="0 0 -0.001",
                          type="plane", material="groundplane", group="1", contype="0", conaffinity="0")
        else:
            gp_geom.set("material", "groundplane")

    temp_xml = os.path.join(env_dir, "_temp_composed_scene.xml")
    env_tree.write(temp_xml)
    try:
        model = mujoco.MjModel.from_xml_path(temp_xml)
        print(f"Successfully loaded indoor scene with drone! Bodies: {model.nbody}, Geoms: {model.ngeom}")
    finally:
        if os.path.exists(temp_xml):
            os.remove(temp_xml)

    return model


def load_model_dynamic(drone_scene_path: str, env_scene_path: str = None, custom_spawn: list = None, no_collision: bool = False) -> (mujoco.MjModel, list):
    """
    Dynamically loads and composes a drone model with an environment scene.
    Returns (model, spawn_position).
    """
    if not env_scene_path or env_scene_path.lower() == "none":
        spawn_pos = custom_spawn if custom_spawn is not None else DEFAULT_SPAWN_POS
        print(f"Loading base drone model (empty scene) from: {drone_scene_path}")
        return mujoco.MjModel.from_xml_path(drone_scene_path), spawn_pos

    if not os.path.exists(env_scene_path):
        print(f"Warning: Environment scene '{env_scene_path}' not found. Falling back to base scene.")
        spawn_pos = custom_spawn if custom_spawn is not None else DEFAULT_SPAWN_POS
        return mujoco.MjModel.from_xml_path(drone_scene_path), spawn_pos

    print(f"Dynamically loading environment: {env_scene_path}")

    # Determine spawn position:
    # 1. Custom CLI override
    # 2. Scene XML site (name="drone_spawn" or contains "spawn")
    # 3. Scene XML keyframe (name="spawn")
    # 4. Fallback based on scene name
    spawn_pos = custom_spawn
    if spawn_pos is None:
        try:
            tree = ET.parse(env_scene_path)
            root = tree.getroot()
            for s in root.findall(".//site"):
                if "spawn" in s.get("name", "").lower():
                    coords = [float(v) for v in s.get("pos", "").split()]
                    if len(coords) == 3:
                        spawn_pos = coords
                        break
            if spawn_pos is None:
                for k in root.findall(".//key"):
                    if "spawn" in k.get("name", "").lower():
                        coords = [float(v) for v in k.get("qpos", "").split()[:3]]
                        if len(coords) == 3:
                            spawn_pos = coords
                            break
        except Exception as e:
            print(f"Note: Could not parse spawn from XML: {e}")

    if spawn_pos is None:
        if "scenesmith" in env_scene_path.lower():
            spawn_pos = OFFICE_SPAWN_POS
        else:
            spawn_pos = DEFAULT_SPAWN_POS

    print(f"Resolved Drone Spawn Position: {spawn_pos}")

    is_indoor_mesh_scene = "scenesmith" in env_scene_path.lower() or os.path.exists(
        os.path.join(os.path.dirname(env_scene_path), "meshes")
    )

    if is_indoor_mesh_scene:
        model = _compose_mesh_scene(env_scene_path, DRONE_BODY_XML, spawn_pos=spawn_pos, no_collision=no_collision)
        return model, spawn_pos

    # Additive scenes (e.g. gates.xml)
    base_tree = ET.parse(drone_scene_path)
    base_root = base_tree.getroot()
    base_worldbody = base_root.find("worldbody")
    if base_worldbody is None:
        base_worldbody = ET.SubElement(base_root, "worldbody")

    env_tree = ET.parse(env_scene_path)
    env_root = env_tree.getroot()
    env_worldbody = env_root.find("worldbody")
    if env_worldbody is not None:
        for child in list(env_worldbody):
            if no_collision and child.tag == "geom":
                child.set("contype", "0")
                child.set("conaffinity", "0")
            base_worldbody.append(child)

    env_asset = env_root.find("asset")
    if env_asset is not None:
        base_asset = base_root.find("asset")
        if base_asset is None:
            base_asset = ET.SubElement(base_root, "asset")
        for child in list(env_asset):
            base_asset.append(child)

    drone_dir = os.path.dirname(os.path.abspath(drone_scene_path))
    temp_combined_path = os.path.join(drone_dir, "_temp_composed_scene.xml")
    base_tree.write(temp_combined_path)

    try:
        model = mujoco.MjModel.from_xml_path(temp_combined_path)
        print(f"Successfully composed model! Total bodies: {model.nbody}, Total geoms: {model.ngeom}")
    finally:
        if os.path.exists(temp_combined_path):
            os.remove(temp_combined_path)

    return model, spawn_pos


def setup_camera(model):
    """Resolve the camera ID, falling back gracefully if necessary."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id != -1:
        print(f"Using camera '{CAMERA_NAME}' (ID: {cam_id})")
        return CAMERA_NAME

    fallback_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    if fallback_id != -1:
        print(f"Camera '{CAMERA_NAME}' not found, falling back to 'fpv' (ID: {fallback_id})")
        return "fpv"

    print("Warning: Specific front camera not found. Using default camera index 0.")
    return 0


def render_rgbd_frame(renderer, data, camera, controller):
    """
    Renders both RGB and Depth from the camera,
    combining them into a compact OpenCV image with flight telemetry overlay.
    """
    # 1. Update scene once for both RGB and Depth passes (cuts rendering overhead)
    renderer.update_scene(data, camera=camera)

    # 2. Render RGB image
    renderer.disable_depth_rendering()
    rgb_frame = renderer.render()
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # 3. Render Depth image (in meters)
    renderer.enable_depth_rendering()
    depth_meters = renderer.render()

    center_y, center_x = CAM_HEIGHT // 2, CAM_WIDTH // 2
    center_dist = depth_meters[center_y, center_x]

    # Normalize depth to 0-255 for TURBO color mapping
    clipped_depth = np.clip(depth_meters, DEPTH_MIN_METERS, DEPTH_MAX_METERS)
    depth_norm = (
        (clipped_depth - DEPTH_MIN_METERS) / (DEPTH_MAX_METERS - DEPTH_MIN_METERS) * 255.0
    ).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

    # Telemetry data
    alt = data.qpos[2]
    speed = np.linalg.norm(data.qvel[0:3])
    speed_kmh = speed * 3.6
    w, x, y, z = data.qpos[3:7]
    _, _, yaw = controller.quat_to_euler(w, x, y, z)
    yaw_deg = int(np.degrees(yaw)) % 360

    # Color highlights
    mode_color = (0, 255, 255) if controller.flight_mode_name == "SPORT" else (255, 255, 0)
    action_color = (0, 255, 0) if "FWD" in controller.action_label or "TURBO" in controller.action_label else (255, 255, 255)

    # Overlay HUD on RGB
    cv2.putText(bgr_frame, f"[{controller.flight_mode_name}] {controller.action_label}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, action_color, 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Alt: {alt:.2f}m", (4, CAM_HEIGHT - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay HUD on Depth
    cv2.putText(depth_colored, f"Depth: {center_dist:.2f}m", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(depth_colored, f"{speed_kmh:.1f}km/h | {yaw_deg}°", (4, CAM_HEIGHT - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)

    # Stack horizontally: [RGB | Depth] -> Total window size = (2 * CAM_WIDTH, CAM_HEIGHT)
    rgbd_combined = np.hstack((bgr_frame, depth_colored))
    return rgbd_combined


def print_controls():
    """Prints the keyboard controls in the terminal."""
    print("\n" + "=" * 78)
    print("🛸 REALISTIC SMOOTH DRONE FLIGHT SIMULATOR (AERODYNAMIC PHYSICS)")
    print("=" * 78)
    print("  [ Throttle & Altitude ] (Left Hand)")
    print("    HOLD  W  /  SPACE  : Power Climb Up (smooth aerodynamic ascent)")
    print("    HOLD  S  /  C      : Fast Descent (locks altitude on release)")
    print("")
    print("  [ Yaw / Heading Steering ] (Left Hand)")
    print("    HOLD  A  /  Q      : Turn Left  (smooth heading rotation CCW)")
    print("    HOLD  D  /  E      : Turn Right (smooth heading rotation CW)")
    print("")
    print("  [ Directional Flight & Tilt ] (Right Hand / Arrows)")
    print("    HOLD  I  /  UP     : Fly Forward (realistic aerodynamic nose-down tilt)")
    print("    HOLD  K  /  DOWN   : Fly Backward (aerodynamic pitch-up tilt)")
    print("    HOLD  J  /  LEFT   : Bank Left (realistic aerodynamic roll left)")
    print("    HOLD  L  /  RIGHT  : Bank Right (realistic aerodynamic roll right)")
    print("    HOLD  SHIFT        : Turbo Sprint (boosts tilt & speed)")
    print("")
    print("  [ Camera Modes & Flight Toggles ]")
    print("    PRESS C            : 🎥 Toggle Camera (Mode 1: Free at Drone -> Mode 2: FPV -> Mode 3: GTA Follow)")
    print("    PRESS M            : Switch Speed (CRUISE <--> SPORT)")
    print("    PRESS B            : Toggle Auto-Braking (ACTIVE BRAKE <--> MANUAL COAST)")
    print("    PRESS H            : Immediate Full Stop (Hover Lock)")
    print("    PRESS R            : Reset Drone to Spawn Position")
    print("    PRESS Q / ESC      : Exit Simulation")
    print("=" * 78)
    print("🔥 Real-time physical key holds detected anywhere (MuJoCo Viewer or OpenCV window)!\n")


def main():
    parser = argparse.ArgumentParser(description="High-Speed Racing Drone Simulator with RGB-D Camera")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to drone scene XML (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=ENV_SCENE_PATH,
        help=f"Environment scene to dynamically load (default: '{ENV_SCENE_PATH}'; use 'none' for empty space)",
    )
    parser.add_argument(
        "--spawn",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Custom drone spawn position [X Y Z] (overrides scene default)",
    )
    parser.add_argument(
        "--no-cam",
        action="store_true",
        help="Disable offscreen RGB-D camera rendering and OpenCV window for maximum simulator framerate",
    )
    parser.add_argument(
        "--cam-fps",
        type=float,
        default=20.0,
        help="Target camera sensor update rate in Hz (default: 20.0; lower values reduce rendering overhead)",
    )
    parser.add_argument(
        "--no-collision",
        "--disable-collisions",
        dest="no_collision",
        action="store_true",
        help="Disable all collision geoms and collision checking for maximum real-time simulation framerate (uses textured visual mesh only).",
    )
    args = parser.parse_args()

    if cv2 is None and not args.no_cam:
        print("[!] OpenCV is not installed; continuing without the optional RGB-D window.")
        print("    Install it with 'pip install opencv-python' to enable that window.")
        args.no_cam = True

    # Load model and resolve spawn point
    model, spawn_pos = load_model_dynamic(
        drone_scene_path=args.model,
        env_scene_path=args.scene,
        custom_spawn=args.spawn,
        no_collision=args.no_collision
    )
    data = mujoco.MjData(model)

    # Initialize continuous hardware keyboard poller & racing controller
    kb = ContinuousKeyboard()
    controller = RacingFlightController()

    def reset_drone():
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.qpos[0] = spawn_pos[0]
        data.qpos[1] = spawn_pos[1]
        data.qpos[2] = spawn_pos[2]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        controller.reset(spawn_pos)

    reset_drone()

    # Resolve camera & offscreen renderer (only if camera enabled)
    camera = None
    renderer = None
    if not args.no_cam:
        camera = setup_camera(model)
        renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        print(f"[*] RGB-D Camera enabled at target {args.cam_fps:.1f} FPS.")
    else:
        print("[*] RGB-D Camera disabled (--no-cam). Running at maximum native framerate.")

    # Camera Mode Switching System:
    # Mode 1: Standard Free Camera (Spawns at Drone Position)
    # Mode 2: Drone FPV Cockpit Camera
    # Mode 3: GTA-Style Third-Person Follow Camera
    CAMERA_MODES = [
        (0, "MODE 1: Standard Free Camera (Centered at Drone)"),
        (1, "MODE 2: Drone FPV Cockpit Camera"),
        (2, "MODE 3: GTA-Style Third-Person Follow Camera"),
    ]
    current_cam_mode = 0
    last_c_press = 0.0

    # Resolve FPV camera ID on drone
    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    if fpv_cam_id == -1:
        fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")

    # Initial follow camera heading
    w, x, y, z = data.qpos[3:7]
    _, _, init_yaw = controller.quat_to_euler(w, x, y, z)
    cam_follow_azimuth = float(np.degrees(init_yaw))
    cam_smooth_lookat = np.array(data.qpos[:3], dtype=np.float64)
    active_viewer = None

    def apply_camera_mode_switch(v):
        nonlocal current_cam_mode, cam_follow_azimuth, cam_smooth_lookat
        if v is None:
            return
        current_cam_mode = (current_cam_mode + 1) % 3
        _, mode_name = CAMERA_MODES[current_cam_mode]
        short_names = ["FREE", "FPV", "GTA"]
        controller.camera_mode_name = short_names[current_cam_mode]
        print(f"\n>>> 🎥 Viewer Camera Switched to: [{mode_name}] <<<")

        w, x, y, z = data.qpos[3:7]
        _, _, cur_yaw = controller.quat_to_euler(w, x, y, z)

        if current_cam_mode == 0:
            # Mode 1: Standard Free Camera (spawns at exact drone coordinates)
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.fixedcamid = -1
            v.cam.lookat[0] = data.qpos[0]
            v.cam.lookat[1] = data.qpos[1]
            v.cam.lookat[2] = data.qpos[2]
            v.cam.distance = 2.5
            v.cam.azimuth = float(np.degrees(cur_yaw))
            v.cam.elevation = -15.0

        elif current_cam_mode == 1:
            # Mode 2: Drone FPV camera
            if fpv_cam_id != -1:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                v.cam.fixedcamid = fpv_cam_id
            else:
                print("Warning: FPV camera not found on drone model.")

        elif current_cam_mode == 2:
            # Mode 3: GTA Follow Camera (STRICTLY behind drone, facing along heading)
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.fixedcamid = -1
            v.cam.lookat[0] = data.qpos[0]
            v.cam.lookat[1] = data.qpos[1]
            v.cam.lookat[2] = data.qpos[2] + 0.15
            v.cam.distance = 2.4
            v.cam.elevation = -12.0
            v.cam.azimuth = float(np.degrees(cur_yaw))

    # Event keys from GLFW / viewer
    last_m_press = 0.0
    last_b_press = 0.0

    def on_mujoco_key(key):
        """Callback for single-action hotkeys in MuJoCo viewer."""
        nonlocal last_m_press, last_b_press, last_c_press
        k = key & 0xFF
        if k in (ord('r'), ord('R'), 82):
            reset_drone()
        elif k in (ord('c'), ord('C'), 67):
            now = time.time()
            if now - last_c_press > 0.35:
                apply_camera_mode_switch(active_viewer)
                last_c_press = now
        elif k in (ord('m'), ord('M'), 77):
            now = time.time()
            if now - last_m_press > 0.3:
                controller.toggle_mode()
                last_m_press = now
        elif k in (ord('b'), ord('B'), 66):
            now = time.time()
            if now - last_b_press > 0.3:
                controller.toggle_autobrake()
                last_b_press = now

    print_controls()

    render_interval = 1.0 / max(1.0, args.cam_fps) if not args.no_cam else 1e9
    last_render_time = 0.0
    last_viewer_sync = 0.0
    dt = model.opt.timestep

    # Fixed-timestep physics accumulator for true 1.0x real-time speed
    accumulator = 0.0
    last_sim_time = time.perf_counter()

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_mujoco_key) as viewer:
            active_viewer = viewer
            # Mode 1 init: spawn camera centered at drone
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.fixedcamid = -1
            viewer.cam.lookat[0] = spawn_pos[0]
            viewer.cam.lookat[1] = spawn_pos[1]
            viewer.cam.lookat[2] = spawn_pos[2]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = cam_follow_azimuth
            viewer.cam.elevation = -15.0

            while viewer.is_running():
                now = time.perf_counter()
                elapsed = now - last_sim_time
                last_sim_time = now

                # Clamp elapsed to prevent spiral-of-death (max 0.15s catchup)
                elapsed = min(elapsed, 0.15)
                accumulator += elapsed

                # Step physics to match elapsed real time exactly (eliminates slow-motion lag)
                stepped = False
                while accumulator >= dt:
                    # Check for single-press toggles
                    if kb.is_down('R'):
                        reset_drone()
                    if kb.is_down('C'):
                        t_now = time.time()
                        if t_now - last_c_press > 0.35:
                            apply_camera_mode_switch(viewer)
                            last_c_press = t_now
                    if kb.is_down('M'):
                        t_now = time.time()
                        if t_now - last_m_press > 0.35:
                            controller.toggle_mode()
                            last_m_press = t_now
                    if kb.is_down('B'):
                        t_now = time.time()
                        if t_now - last_b_press > 0.35:
                            controller.toggle_autobrake()
                            last_b_press = t_now

                    # Update flight targets directly from continuous physical hardware keys
                    controller.update_from_keyboard(kb, dt)

                    # Compute racing flight forces and step physics
                    data.ctrl[:] = controller.compute_control(data, dt)
                    mujoco.mj_step(model, data)
                    accumulator -= dt
                    stepped = True

                # Mode 3: GTA Follow Camera continuous tracking (STRICTLY behind drone, facing along heading)
                if current_cam_mode == 2:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.fixedcamid = -1
                    viewer.cam.lookat[0] = data.qpos[0]
                    viewer.cam.lookat[1] = data.qpos[1]
                    viewer.cam.lookat[2] = data.qpos[2] + 0.15
                    viewer.cam.distance = 2.4
                    viewer.cam.elevation = -12.0

                    w, x, y, z = data.qpos[3:7]
                    _, _, drone_yaw = controller.quat_to_euler(w, x, y, z)
                    viewer.cam.azimuth = float(np.degrees(drone_yaw))

                # Sync viewer at 60 FPS
                if stepped and (now - last_viewer_sync >= 0.016):
                    viewer.sync()
                    last_viewer_sync = now

                # Yield brief time slice to OS/renderer when idle to maintain high framerate without CPU churn
                if not stepped:
                    time.sleep(0.0005)

                # Render camera feed at capped cam_fps for high performance
                if not args.no_cam and (now - last_render_time >= render_interval):
                    rgbd_img = render_rgbd_frame(renderer, data, camera, controller)
                    cv2.imshow(WINDOW_NAME, rgbd_img)
                    last_render_time = now

                    # Process keys from OpenCV window
                    cv_key = cv2.waitKeyEx(1)
                    if cv_key > 0:
                        k = cv_key & 0xFF
                        if k in (27, ord('q'), ord('Q')):
                            print("Quit signal received.")
                            break
                        elif k in (ord('r'), ord('R')):
                            reset_drone()
                        elif k in (ord('c'), ord('C')):
                            t_now = time.time()
                            if t_now - last_c_press > 0.35:
                                apply_camera_mode_switch(viewer)
                                last_c_press = t_now
                        elif k in (ord('m'), ord('M')):
                            t_now = time.time()
                            if t_now - last_m_press > 0.35:
                                controller.toggle_mode()
                                last_m_press = t_now
                        elif k in (ord('b'), ord('B')):
                            t_now = time.time()
                            if t_now - last_b_press > 0.35:
                                controller.toggle_autobrake()
                                last_b_press = t_now
                elif args.no_cam:
                    # Give CPU brief yield when running without offscreen render overhead
                    time.sleep(0.001)

    except Exception as e:
        print(f"Viewer launch not available or running headless: {e}")
        if not args.no_cam and renderer is not None:
            print("Running headlessly with OpenCV camera window...")
            for i in range(1000):
                controller.update_from_keyboard(kb, dt)
                data.ctrl[:] = controller.compute_control(data, dt)
                mujoco.mj_step(model, data)
                if i % max(1, int(1.0 / (args.cam_fps * model.opt.timestep))) == 0:
                    rgbd_img = render_rgbd_frame(renderer, data, camera, controller)
                    cv2.imshow(WINDOW_NAME, rgbd_img)
                    key = cv2.waitKeyEx(1)
                    if key > 0 and (key & 0xFF in (27, ord('q'))):
                        break
            print("Completed headless run.")
    finally:
        if renderer is not None:
            renderer.close()
        if not args.no_cam:
            cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
