# pyright: reportAttributeAccessIssue=false
# (mujoco assembles its namespace dynamically, which static analysis cannot follow)
"""Operational-space controller: turns EE pose targets into the sim's 9-D normalized joint-velocity action.

All quantities are read from the current (noise-free) sim state, so the action is a pure function of the
state (Markovian). Arm joints use damped-least-squares IK; the fingers use a saturating P controller.
"""

from typing import Literal

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from markovian_policy.arrays import FloatArray
from markovian_policy.sim.actuation import ACT_AMP
from markovian_policy.sim.physics import KitchenPhysics

type GripperTarget = Literal["open", "close"] | float  # or a per-finger opening in [0, 0.04]

# --- env-specific constants -------------------------------------------------
EE_SITE = "end_effector"  # site on panda0_link7, ~at the fingertips
N_ARM = 7  # arm joints occupy qpos/qvel/jac cols 0..6
GRIPPER_QPOS = (7, 8)  # finger_joint1, finger_joint2 in qpos
FINGER_OPEN = 0.04  # per-finger joint upper limit (fully open)
FINGER_CLOSED = 0.0  # per-finger joint lower limit (closed)

# control gains / limits
KP_POS = 8.0  # cartesian P gain (1/s) on position error
KP_ROT = 3.0  # cartesian P gain (1/s) on orientation error
V_LIN_MAX = 0.4  # m/s cap on commanded linear EE velocity
V_ANG_MAX = 2.0  # rad/s cap on commanded angular EE velocity
DLS_LAMBDA = 0.1  # damped-least-squares damping (rad)
# Large enough that a "close"/"open" command saturates the finger velocity and
# the position-actuator target clips to its limit (0 or 0.04). That matters for
# grasping: the env's velocity actuation recomputes the finger target from the current
# (blocked) finger position each step, so a gentle gain leaves only a tiny
# position error and a weak squeeze (~6 N) that slips; saturating clips the
# target to 0 and gives the full ~10 N needed to hold the handle under load.
KP_GRIP = 25.0  # P gain (1/s) driving fingers to their target opening
KP_JOINT = 4.0  # joint-space P gain (1/s) on joint position error
V_JOINT_MAX = 1.5  # rad/s cap per arm joint, used by joint_pose_to_action


def ee_pos(sim: KitchenPhysics) -> FloatArray:
    """World position (3,) of the end-effector site."""
    return sim.data.site(EE_SITE).xpos.copy()


def ee_rot(sim: KitchenPhysics) -> FloatArray:
    """World rotation matrix (3, 3) of the end-effector site."""
    return sim.data.site(EE_SITE).xmat.reshape(3, 3).copy()


def gripper_width(sim: KitchenPhysics) -> float:
    """Sum of the two finger openings (0 closed .. ~0.08 fully open)."""
    return float(sim.data.qpos[GRIPPER_QPOS[0]] + sim.data.qpos[GRIPPER_QPOS[1]])


def body_point_world(sim: KitchenPhysics, body_name: str, local_offset: ArrayLike = (0.0, 0.0, 0.0)) -> FloatArray:
    """World position of a point given in a body's local frame."""
    body = sim.data.body(body_name)
    return body.xpos + body.xmat.reshape(3, 3) @ np.asarray(local_offset, dtype=float)


def joint_anchor_axis(sim: KitchenPhysics, joint_name: str) -> tuple[FloatArray, FloatArray]:
    """World-frame (anchor, axis) of a hinge joint, each (3,)."""
    joint = sim.data.joint(joint_name)
    return joint.xanchor.copy(), joint.xaxis.copy()


def site_jacobian(sim: KitchenPhysics) -> tuple[FloatArray, FloatArray]:
    """Linear (3, 7) and angular (3, 7) EE site Jacobians over the arm joints."""
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, sim.data.site(EE_SITE).id)
    return jacp[:, :N_ARM], jacr[:, :N_ARM]


def gravity_feedforward(sim: KitchenPhysics) -> FloatArray:
    """Joint-velocity feedforward (N_ARM,) that holds the arm against gravity.

    The sim integrates the action (a joint *velocity*) into the position-actuator
    setpoint once per control step: ctrl = qpos + vel*dt. A MuJoCo position
    actuator supplies torque kp*(ctrl - qpos), so to balance the gravity/bias
    torque tau (= data.qfrc_bias, gravity + Coriolis) it needs a standing offset
    ctrl - qpos = tau/kp. Feeding vel_ff = tau/(kp*dt) produces exactly that
    offset, so the cartesian P term no longer has to leave a residual pose error
    to supply the holding torque -- that residual *is* the steady-state droop,
    worst at high/extended reach where the weak wrist actuators (kp=120) near
    saturate. tau is a function of the current state, so this stays Markovian.
    """
    kp = sim.model.actuator_gainprm[:N_ARM, 0]
    return sim.data.qfrc_bias[:N_ARM] / (kp * sim.dt)


def orientation_error(R_cur: np.ndarray, R_des: np.ndarray) -> FloatArray:
    """Angular error (3,) in world frame that rotates R_cur toward R_des.

    Geometric axis-angle error: the rotation vector of R_des @ R_cur^T. Unlike
    the small-angle sum-of-cross-products form, this stays accurate for large
    targets (e.g. the 45 deg open-phase yaw), so the commanded angular velocity
    points along the true error axis and the EE doesn't pitch out of plane.
    """
    R_err = R_des @ R_cur.T
    w = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    s = np.linalg.norm(w) / 2.0  # sin(angle)
    c = (np.trace(R_err) - 1.0) / 2.0  # cos(angle)
    if s < 1e-6:
        return 0.5 * w  # small angle: w/2 ~= angle * axis
    angle = np.arctan2(s, c)
    return (angle / (2.0 * s)) * w  # = angle * unit_axis


def pose_to_action(
    sim: KitchenPhysics,
    target_pos: ArrayLike,
    target_rot: ArrayLike | None = None,
    gripper: GripperTarget = "open",
    gravity_comp: bool = True,
) -> FloatArray:
    """Map a desired EE pose + gripper command to the 9-D normalized action.

    Args:
        target_pos: (3,) desired world position of the EE site.
        target_rot: optional (3, 3) desired world rotation of the EE site.
                    If None, orientation is left uncontrolled.
        gripper: "open" or "close" (or a float per-finger target in [0, 0.04]).
        gravity_comp: add a joint-velocity feedforward that holds the arm against
                    gravity (see gravity_feedforward), removing the steady-state
                    droop of the pure-P cartesian loop. On by default.

    Returns:
        (9,) action in [-1, 1] suitable for sim.step.
    """
    # --- cartesian twist from pose error ---
    pos_err = np.asarray(target_pos, dtype=float) - ee_pos(sim)
    v_lin = np.clip(KP_POS * pos_err, -V_LIN_MAX, V_LIN_MAX)

    jacp, jacr = site_jacobian(sim)
    if target_rot is not None:
        rot_err = orientation_error(ee_rot(sim), np.asarray(target_rot, float))
        v_ang = np.clip(KP_ROT * rot_err, -V_ANG_MAX, V_ANG_MAX)
        jacobian = np.vstack([jacp, jacr])  # 6x7
        twist = np.concatenate([v_lin, v_ang])
    else:
        jacobian = jacp  # 3x7
        twist = v_lin

    # --- damped least squares: dq = jacobian^T (jacobian jacobian^T + lam^2 I)^-1 twist ---
    m = jacobian.shape[0]
    JJt = jacobian @ jacobian.T + (DLS_LAMBDA**2) * np.eye(m)
    dq_arm = jacobian.T @ np.linalg.solve(JJt, twist)  # (7,)

    # --- gravity-compensation feedforward (holds the arm, kills steady-state droop) ---
    if gravity_comp:
        dq_arm = dq_arm + gravity_feedforward(sim)

    dq_grip = _gripper_dq(sim, gripper)
    qvel_cmd = np.concatenate([dq_arm, dq_grip])  # (9,)
    action = np.clip(qvel_cmd / ACT_AMP, -1.0, 1.0)
    return action


def _gripper_dq(sim: KitchenPhysics, gripper: GripperTarget) -> FloatArray:
    """Finger joint velocities (2,) driving toward the gripper command."""
    if gripper == "open":
        finger_target = FINGER_OPEN
    elif gripper == "close":
        finger_target = FINGER_CLOSED
    else:
        finger_target = float(gripper)
    qp = sim.data.qpos
    return np.array(
        [
            KP_GRIP * (finger_target - qp[GRIPPER_QPOS[0]]),
            KP_GRIP * (finger_target - qp[GRIPPER_QPOS[1]]),
        ]
    )


def joint_pose_to_action(sim: KitchenPhysics, target_arm_qpos: ArrayLike, gripper: GripperTarget, gravity_comp: bool = True) -> FloatArray:
    """Map a desired arm joint configuration + gripper command to the 9-D action.

    Drives the 7 arm joints directly in joint space (P control on joint
    position error) rather than through the Cartesian operational-space
    controller. Matching EE pose alone leaves the arm's redundant null-space
    configuration unconstrained -- two subtasks can both call themselves "home"
    at the same EE pose via different joint configurations, one of which may be
    collision-prone for the next subtask's approach. Driving to a specific joint
    configuration (the sim's reset qpos) removes that ambiguity.
    """
    qp = sim.data.qpos
    q_err = np.asarray(target_arm_qpos, dtype=float) - qp[:N_ARM]
    dq_arm = np.clip(KP_JOINT * q_err, -V_JOINT_MAX, V_JOINT_MAX)

    if gravity_comp:
        dq_arm = dq_arm + gravity_feedforward(sim)

    dq_grip = _gripper_dq(sim, gripper)
    qvel_cmd = np.concatenate([dq_arm, dq_grip])  # (9,)
    action = np.clip(qvel_cmd / ACT_AMP, -1.0, 1.0)
    return action
