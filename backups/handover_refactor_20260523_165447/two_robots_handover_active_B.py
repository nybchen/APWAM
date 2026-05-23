import numpy as np

from robofactory.tasks import TwoRobotsHandoverActiveBEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
from robofactory.planner.solutions.two_robots_handover_active_A import (
    HANDOVER_CENTER,
    HANDOVER_RETREAT_DIST,
    _move,
    _move_joint_pose_sequence,
    _move_pose_sequence,
    _position_mid_pose,
    _pre_side_grasp_pose,
    _pre_top_grasp_pose,
    _y_offset_pose,
)


AGENT0_PERCEPTION_POSE = np.array(
    [0.217725, 0.0798748, 0.245282, 0.52818327, -0.35968308, -0.72184916, -0.26567706]
)
AGENT0_HANDOVER_READY_POSE = np.array(
    [0.12493, -0.0787286, 0.346096, 0.52818327, -0.35968308, -0.72184916, -0.26567706]
)
AGENT0_DROP_POSE = np.array(
    [0.0, -0.285, 0.151, 0.095863409, -0.73314507, -0.67211006, -0.039706204]
)
AGENT1_POST_HANDOVER_POSE = np.array(
    [0.0, -0.039336, 0.26, 0.36652983, 0.60535673, -0.60446073, 0.36582283]
)


def _agent1_side_handover_pose(env):
    approaching = np.array([0.0, -1.0, 0.0])
    closing = np.array([-1.0, 0.0, 0.0])
    half_size = float(env.cube_half_size_value)
    finger_depth = 0.025
    tcp_center = HANDOVER_CENTER + approaching * (
        -half_size + min(finger_depth, half_size)
    )
    grasp_pose = env.agent.agents[1].build_grasp_pose(
        approaching, closing, tcp_center
    )
    return np.array(list(grasp_pose.p) + list(grasp_pose.q))


def _agent0_vertical_handover_pose(env):
    approaching = np.array([0.0, 1.0, 0.0])
    closing = np.array([0.0, 0.0, -1.0])
    half_size = float(env.cube_half_size_value)
    finger_depth = 0.025
    tcp_center = HANDOVER_CENTER + approaching * (
        -half_size + min(finger_depth, half_size)
    )
    grasp_pose = env.agent.agents[0].build_grasp_pose(
        approaching, closing, tcp_center
    )
    return np.array(list(grasp_pose.p) + list(grasp_pose.q))


def _pick_other_cube_sides(pose):
    other_sides_pose = pose.copy()
    w, x, y, z = other_sides_pose[3:]
    c = np.sqrt(0.5)
    s = np.sqrt(0.5)
    other_sides_pose[3:] = np.array(
        [
            w * c - z * s,
            x * c + y * s,
            y * c - x * s,
            w * s + z * c,
        ]
    )
    return other_sides_pose


def solve(env: TwoRobotsHandoverActiveBEnv, seed=None, debug=False, vis=False, reset=True):
    """
    Right-to-left mirror of A.

    Agent 1 starts at the right goal and hands the cube to agent 0. Agent 0
    observes first, receives with a top/bottom clamp, then drops at the left
    goal while agent 1 mirrors agent 0's post-handover observation role in A.
    """
    if reset:
        env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True,
    )
    env = env.unwrapped

    planner.set_mode_label("perception")
    if not _move(
        planner,
        AGENT0_PERCEPTION_POSE,
        move_id=0,
        mode_label="perception",
        name="agent0 initial perception pose",
    ):
        planner.close()
        return -1

    pick_pose = _pick_other_cube_sides(planner.get_grasp_pose_from_obb(env.cube, 1))
    pre_pick_pose = _pre_top_grasp_pose(pick_pose)

    planner.set_mode_label("perception")
    if not _move(planner, pre_pick_pose, move_id=1, name="agent1 pre-pick"):
        planner.close()
        return -1
    if not _move(planner, pick_pose, move_id=1, name="agent1 pick"):
        planner.close()
        return -1
    planner.close_gripper(1)

    handover_pose_agent1 = _agent1_side_handover_pose(env)
    pre_handover_pose_agent1 = _pre_side_grasp_pose(
        handover_pose_agent1, side="positive_y"
    )
    agent0_ready_mid_pose = _position_mid_pose(
        AGENT0_PERCEPTION_POSE, AGENT0_HANDOVER_READY_POSE
    )
    if not _move_joint_pose_sequence(
        planner,
        [
            (agent0_ready_mid_pose, pre_handover_pose_agent1),
            (AGENT0_HANDOVER_READY_POSE, handover_pose_agent1),
        ],
        mode_label=["perception", "action"],
        names=[
            "agent1 lift to pre-handover while agent0 raises",
            "agent1 enter handover while agent0 reaches ready pose",
        ],
    ):
        planner.close()
        return -1

    handover_pose_agent0 = _agent0_vertical_handover_pose(env)
    pre_handover_pose_agent0 = _pre_side_grasp_pose(
        handover_pose_agent0, side="negative_y"
    )
    if not _move_pose_sequence(
        planner,
        [
            pre_handover_pose_agent0,
            handover_pose_agent0,
        ],
        move_id=0,
        names=["agent0 pre-handover approach", "agent0 top-bottom handover grasp"],
    ):
        planner.close()
        return -1

    planner.set_mode_label("action", agent_ids=[0, 1])
    planner.close_gripper(0)
    planner.open_gripper(1)

    pre_drop_pose = _pre_top_grasp_pose(AGENT0_DROP_POSE, distance=0.12)
    retreat_pose_agent0 = _y_offset_pose(handover_pose_agent0, -HANDOVER_RETREAT_DIST)
    retreat_pose_agent1 = _y_offset_pose(handover_pose_agent1, HANDOVER_RETREAT_DIST)
    if not _move_joint_pose_sequence(
        planner,
        [
            (retreat_pose_agent0, retreat_pose_agent1),
            (pre_drop_pose, AGENT1_POST_HANDOVER_POSE),
        ],
        mode_label=["action", "perception"],
        names=[
            "both agents retreat after handover",
            "agent1 post-handover observation while agent0 carries to pre-drop",
        ],
    ):
        planner.close()
        return -1

    if not _move_joint_pose_sequence(
        planner,
        [
            (AGENT0_DROP_POSE, AGENT1_POST_HANDOVER_POSE),
        ],
        mode_label=["action", "perception"],
        names=[
            "agent0 drops cube while agent1 watches",
        ],
    ):
        planner.close()
        return -1
    res = planner.open_gripper(0)

    planner.close()
    return res
