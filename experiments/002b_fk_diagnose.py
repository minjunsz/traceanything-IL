"""Step-by-step diagnostic for the franka-kitchen env silently exiting(1).

adept_envs/simulation/module.py catches ImportError around `import mujoco_py`
with a bare `except ImportError: print(generic); sys.exit(1)`, which can mask
the real error. This script imports each layer separately, with faulthandler
enabled, so a hard crash (segfault, C-level exit()) or a real traceback
surfaces instead of vanishing.
"""

import faulthandler
import os

faulthandler.enable()

print("DISPLAY =", os.environ.get("DISPLAY"), flush=True)
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"), flush=True)

print("\n[1] import mujoco_py", flush=True)
import mujoco_py

print("    OK:", mujoco_py.__file__, flush=True)

print("\n[2] mujoco_py.builder.get_nvidia_lib_dir()", flush=True)
from mujoco_py.builder import get_nvidia_lib_dir

print("    ->", get_nvidia_lib_dir(), flush=True)

print("\n[3] import glfw (pypi ctypes wrapper, NOT mujoco_py's own)", flush=True)
import glfw

print("    OK:", glfw.__file__, flush=True)

print("\n[4] mujoco_py.load_model_from_path + MjSim on the kitchen XML", flush=True)
model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "third_party",
    "relay-policy-learning",
    "adept_models",
    "kitchen",
    "kitchen.xml",
)
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
print("    OK:", sim, flush=True)

print("\n[5] MjRenderContextOffscreen (this is the actual GL/EGL context)", flush=True)
ctx = mujoco_py.MjRenderContextOffscreen(sim, device_id=0)
print("    OK:", ctx, flush=True)

print("\n[6] render + read_pixels", flush=True)
ctx.render(320, 240, camera_id=-1)
img, _ = ctx.read_pixels(320, 240, depth=True)
print("    OK, image shape:", img.shape, flush=True)

print("\n[7] import dm_control.mujoco (its OWN separate GL backend picker --", flush=True)
print("    respects MUJOCO_GL, unrelated to mujoco_py's get_nvidia_lib_dir)", flush=True)
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"), flush=True)
from dm_control import mujoco as dm_mujoco

print("    OK:", dm_mujoco.__file__, flush=True)

print("\n[8] dm_control Physics + Camera render (the actual path adept_envs/", flush=True)
print("    simulation/renderer.py uses, NOT mujoco_py's MjRenderContextOffscreen)", flush=True)
physics = dm_mujoco.Physics.from_xml_path(model_path)
camera = dm_mujoco.Camera(physics, height=240, width=320, camera_id=-1)
dm_img = camera.render()
print("    OK, image shape:", dm_img.shape, flush=True)

print("\nALL STAGES PASSED", flush=True)
