"""
default_assets.py - Default Sky & Ground Assets for Geo2Sim MuJoCo Environments
================================================================================
Provides realistic default environment assets:
  - Sky: Realistic 360-degree Twilight Sunset Skybox (Pastel Pink, Violet & Azure clouds)
  - Ground Options:
      1. 'reflective' [DEFAULT]: High-reflectance silver-pearl grid surface reflecting the skybox clouds
      2. 'traces': Tron-style pure glowing cyan & orange neon circuit traces
      3. 'chip': Tron-style integrated CPU processor chip array with bus interconnects

Available for simple quick import in Python or direct inclusion in MuJoCo MJCF XML.

Usage in Python:
    import default_assets
    model, data = default_assets.load_default_model()

    # Switch ground style dynamically:
    default_assets.set_ground_style("reflective")  # Default glossy reflective surface
    default_assets.set_ground_style("traces")      # Neon circuit traces
    default_assets.set_ground_style("chip")        # CPU chip array

Usage in MuJoCo MJCF XML:
    <include file="Scenes/default_assets/default_sky_ground.xml"/>
"""

import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco

# Base Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_ROOT / "Scenes" / "default_assets"

# Asset Paths
DEFAULT_SKY_GROUND_XML = ASSETS_DIR / "default_sky_ground.xml"
GROUND_REFLECTIVE_TEXTURE = ASSETS_DIR / "ground_reflective_grid.png"  # Default: Glossy reflective grid
GROUND_TRON_TRACES_TEXTURE = ASSETS_DIR / "ground_tron_traces.png"    # Style: Pure neon circuit traces
GROUND_TRON_CHIP_TEXTURE = ASSETS_DIR / "ground_tron_chip.png"        # Style: Processor & bus array
GROUND_ACTIVE_TEXTURE = ASSETS_DIR / "ground_tron.png"               # Active / backward-compatible texture
SKY_TWILIGHT_TEXTURE = ASSETS_DIR / "sky_twilight.png"

# Default Skydio X2 Drone Scene with Default Assets
DEFAULT_DRONE_SCENE_XML = PACKAGE_ROOT / "mujoco_menagerie" / "skydio_x2" / "scene.xml"

# Quick MJCF Snippet for copy-paste or programmatic injection
MJCF_INCLUDE_SNIPPET = f'<include file="{DEFAULT_SKY_GROUND_XML.as_posix()}"/>'


def get_include_snippet(relative_to: str = None) -> str:
    """Returns the <include file="..."/> tag for MJCF, optionally relative to another directory."""
    if relative_to is not None:
        rel_path = os.path.relpath(DEFAULT_SKY_GROUND_XML, relative_to).replace("\\", "/")
        return f'<include file="{rel_path}"/>'
    return f'<include file="{DEFAULT_SKY_GROUND_XML.as_posix()}"/>'


def load_default_model(scene_path: str = None) -> (mujoco.MjModel, mujoco.MjData):
    """
    Loads and compiles the default drone scene with active ground and Twilight skybox.
    Returns (model, data).
    """
    path_to_load = str(DEFAULT_DRONE_SCENE_XML) if scene_path is None else scene_path
    model = mujoco.MjModel.from_xml_path(path_to_load)
    data = mujoco.MjData(model)
    return model, data


def apply_default_sky_and_ground(xml_tree_or_root, ground_style: str = "reflective"):
    """
    Injects or overrides skybox and groundplane in an existing XML ElementTree
    with the chosen ground texture and Twilight sunset skybox.
    """
    root = xml_tree_or_root.getroot() if hasattr(xml_tree_or_root, "getroot") else xml_tree_or_root

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Update or add skybox
    skybox = asset.find(".//texture[@type='skybox']")
    if skybox is None:
        skybox = ET.SubElement(asset, "texture", type="skybox")
    skybox.set("file", SKY_TWILIGHT_TEXTURE.as_posix())
    skybox.set("gridsize", "3 4")
    skybox.set("gridlayout", ".U..LFRB.D..")

    # Determine ground texture and material parameters based on style
    st = ground_style.lower()
    if st in ("reflective", "grid", "mirror", "default"):
        tex_file = GROUND_REFLECTIVE_TEXTURE.as_posix()
        texrepeat = "3 3"
        reflectance = "0.42"
        specular = "0.65"
        shininess = "0.85"
    elif st == "chip":
        tex_file = GROUND_TRON_CHIP_TEXTURE.as_posix()
        texrepeat = "6 6"
        reflectance = "0.22"
        specular = "0.45"
        shininess = "0.75"
    else:  # 'traces'
        tex_file = GROUND_TRON_TRACES_TEXTURE.as_posix()
        texrepeat = "6 6"
        reflectance = "0.22"
        specular = "0.45"
        shininess = "0.75"

    # Update or add groundplane texture
    gp_tex = asset.find(".//texture[@name='groundplane']")
    if gp_tex is None:
        gp_tex = ET.SubElement(asset, "texture", name="groundplane", type="2d")
    gp_tex.set("file", tex_file)

    # Update or add groundplane material
    gp_mat = asset.find(".//material[@name='groundplane']")
    if gp_mat is None:
        gp_mat = ET.SubElement(asset, "material", name="groundplane")
    gp_mat.set("texture", "groundplane")
    gp_mat.set("texuniform", "true")
    gp_mat.set("texrepeat", texrepeat)
    gp_mat.set("reflectance", reflectance)
    gp_mat.set("specular", specular)
    gp_mat.set("shininess", shininess)

    return root


def set_ground_style(style: str = "reflective"):
    """
    Sets the active default ground texture and updates default_sky_ground.xml:
      - 'reflective': High-reflectance silver-pearl grid surface reflecting sunset clouds [DEFAULT]
      - 'traces': Pure consistent glowing neon circuit traces
      - 'chip': Motherboard processor & bus trace array
    """
    st = style.lower()
    if st in ("reflective", "grid", "mirror", "default"):
        target_tex = GROUND_REFLECTIVE_TEXTURE
        target_name = "ground_reflective_grid.png"
        texrepeat = "3 3"
        reflectance = "0.42"
        specular = "0.65"
        shininess = "0.85"
    elif st == "chip":
        target_tex = GROUND_TRON_CHIP_TEXTURE
        target_name = "ground_tron_chip.png"
        texrepeat = "6 6"
        reflectance = "0.22"
        specular = "0.45"
        shininess = "0.75"
    else:  # 'traces'
        target_tex = GROUND_TRON_TRACES_TEXTURE
        target_name = "ground_tron_traces.png"
        texrepeat = "6 6"
        reflectance = "0.22"
        specular = "0.45"
        shininess = "0.75"

    # Keep ground_tron.png aligned with active texture for full backwards-compatibility
    shutil.copy(target_tex, GROUND_ACTIVE_TEXTURE)

    # Synchronize to Skydio X2 assets directory
    drone_assets_dir = PACKAGE_ROOT / "mujoco_menagerie" / "skydio_x2" / "assets"
    if drone_assets_dir.exists():
        shutil.copy(target_tex, drone_assets_dir / target_name)
        shutil.copy(target_tex, drone_assets_dir / "ground_tron.png")

    # Update default_sky_ground.xml
    xml_content = f"""<mujoco>
  <visual>
    <headlight ambient="0.40 0.40 0.42" diffuse="0.75 0.72 0.78" specular="0.30 0.30 0.35"/>
    <rgba haze="0.48 0.52 0.68 1"/>
    <global offwidth="1920" offheight="1080" azimuth="-25" elevation="-20" ellipsoidinertia="true"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <!-- Realistic Twilight Dusk/Sunset Skybox (Pastel Pink, Violet & Azure Clouds) -->
    <texture type="skybox" file="sky_twilight.png" gridsize="3 4" gridlayout=".U..LFRB.D.."/>

    <!-- Active Ground Plane Texture & Material ({st.upper()}) -->
    <texture type="2d" name="groundplane" file="{target_name}"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="{texrepeat}" reflectance="{reflectance}" specular="{specular}" shininess="{shininess}"/>
  </asset>
</mujoco>
"""
    with open(DEFAULT_SKY_GROUND_XML, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"Active ground style updated to: {st.upper()} ({target_name})")
