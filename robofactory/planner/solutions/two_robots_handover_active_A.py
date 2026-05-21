import numpy as np

from robofactory.tasks import TwoRobotsHandoverActiveAEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


AGENT0_PERCEPTION_POSE = np.array(
    [-0.930056, -0.419985, 0.47093, 0.990189, -0.00493824, 0.0362513, 0.134862]
)
AGENT1_PERCEPTION_POSE = np.array(
    [0.00119789, -0.0493949, 0.143102, 0.41325, 0.578902, 0.561516, -0.422844]
)
HANDOVER_CENTER = np.array([0.0, 0.0, 0.26])
PRE_APPROACH_DIST = 0.14
STAGING_LIFT = 0.08
POST_PICK_LIFT = 0.12


def _top_grasp_pose(env, agent_id: int, cube_center):
    approaching = np.array([0.0, 0.0, -1.0])
    closing = np.array([1.0, 0.0, 0.0])
    half_size = float(env.cube_half_size_value)
    finger_depth = 0.025
    tcp_center = np.array(cube_center, dtype=float)
    tcp_center = tcp_center + approaching * (-half_size + min(finger_depth, half_size))
    grasp_pose = env.agent.agents[agent_id].build_grasp_pose(
        approaching, closing, tcp_center
    )
    return np.array(list(grasp_pose.p) + list(grasp_pose.q))


def _pre_top_grasp_pose(grasp_pose, distance=PRE_APPROACH_DIST):
    pre_pose = grasp_pose.copy()
    pre_pose[2] += distance
    return pre_pose


def _side_grasp_pose(env, agent_id: int, cube_center, side: str):
    if side == "negative_y":
        approaching = np.array([0.0, 1.0, 0.0])
        closing = np.array([1.0, 0.0, 0.0])
    elif side == "positive_y":
        approaching = np.array([0.0, -1.0, 0.0])
        closing = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown side {side!r}")

    half_size = float(env.cube_half_size_value)
    finger_depth = 0.025
    tcp_center = np.array(cube_center, dtype=float)
    tcp_center = tcp_center + approaching * (-half_size + min(finger_depth, half_size))
    grasp_pose = env.agent.agents[agent_id].build_grasp_pose(
        approaching, closing, tcp_center
    )
    return np.array(list(grasp_pose.p) + list(grasp_pose.q))


def _vertical_clamp_pose(env, agent_id: int, cube_center, side: str):
    if side == "negative_y":
        approaching = np.array([0.0, 1.0, 0.0])
    elif side == "positive_y":
        approaching = np.array([0.0, -1.0, 0.0])
    else:
        raise ValueError(f"Unknown side {side!r}")

    closing = np.array([0.0, 0.0, 1.0])
    half_size = float(env.cube_half_size_value)
    finger_depth = 0.025
    tcp_center = np.array(cube_center, dtype=float)
    tcp_center = tcp_center + approaching * (-half_size + min(finger_depth, half_size))
    grasp_pose = env.agent.agents[agent_id].build_grasp_pose(
        approaching, closing, tcp_center
    )
    return np.array(list(grasp_pose.p) + list(grasp_pose.q))


def _pre_side_grasp_pose(grasp_pose, side: str, distance=PRE_APPROACH_DIST):
    pre_pose = grasp_pose.copy()
    if side == "negative_y":
        pre_pose[1] -= distance
    elif side == "positive_y":
        pre_pose[1] += distance
    else:
        raise ValueError(f"Unknown side {side!r}")
    return pre_pose


def _staging_pose(pre_pose, lift=STAGING_LIFT):
    staging_pose = pre_pose.copy()
    staging_pose[2] += lift
    return staging_pose


def _move(planner, pose, move_id, mode_label="action"):
    res = planner.move_to_pose_with_screw(pose, move_id=move_id, mode_label=mode_label)
    return res != -1


def _move_pose_sequence(planner, poses, move_id, mode_label="action"):
    for pose in poses:
        if not _move(planner, pose, move_id=move_id, mode_label=mode_label):
            return False
    return True


def _position_mid_pose(start_pose, target_pose):
    mid_pose = target_pose.copy()
    mid_pose[:3] = 0.5 * (start_pose[:3] + target_pose[:3])
    return mid_pose


def solve(env: TwoRobotsHandoverActiveAEnv, seed=None, debug=False, vis=False):
    """
    Left-to-right handover with explicit opposing gripper geometry.

    Agent 0 picks the cube from above, then rotates into a side-facing
    handover pose and holds the cube's side faces. Agent 1 approaches from
    the opposite side with vertical closing, so it clamps the cube from top
    and bottom during transfer.
    """
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

    # Segment 1: agent 1 watches while agent 0 picks from the left goal region.
    planner.set_mode_label("perception")
    if not _move(planner, AGENT1_PERCEPTION_POSE, move_id=1, mode_label="perception"):
        planner.close()
        return -1

    cube_center = env.cube.pose.p[0].cpu().numpy()
    pick_pose = _top_grasp_pose(env, agent_id=0, cube_center=cube_center)
    pre_pick_pose = _pre_top_grasp_pose(pick_pose)

    planner.set_mode_label("perception")
    if not _move(planner, pre_pick_pose, move_id=0):
        planner.close()
        return -1
    if not _move(planner, pick_pose, move_id=0):
        planner.close()
        return -1
    planner.close_gripper(0)

    post_pick_lift_pose = pick_pose.copy()
    post_pick_lift_pose[2] += POST_PICK_LIFT
    if not _move(planner, post_pick_lift_pose, move_id=0):
        planner.close()
        return -1

    handover_pose_agent0 = _side_grasp_pose(
        env, agent_id=0, cube_center=HANDOVER_CENTER, side="negative_y"
    )
    pre_handover_pose_agent0 = _pre_side_grasp_pose(
        handover_pose_agent0, side="negative_y"
    )
    staging_handover_pose_agent0 = _staging_pose(pre_handover_pose_agent0)
    if not _move_pose_sequence(
        planner,
        [
            staging_handover_pose_agent0,
            pre_handover_pose_agent0,
            handover_pose_agent0,
        ],
        move_id=0,
    ):
        planner.close()
        return -1

    # Segment 2: agent 1 approaches from the opposite side, but its gripper
    # closes along z. Agent 0 holds side faces; agent 1 clamps top/bottom.
    handover_pose_agent1 = _vertical_clamp_pose(
        env, agent_id=1, cube_center=HANDOVER_CENTER, side="positive_y"
    )
    pre_handover_pose_agent1 = _pre_side_grasp_pose(
        handover_pose_agent1, side="positive_y"
    )
    staging_handover_pose_agent1 = _staging_pose(pre_handover_pose_agent1)
    if not _move_pose_sequence(
        planner,
        [
            staging_handover_pose_agent1,
            pre_handover_pose_agent1,
            handover_pose_agent1,
        ],
        move_id=1,
    ):
        planner.close()
        return -1

    planner.set_mode_label("action", agent_ids=[0, 1])
    planner.close_gripper(1)
    planner.open_gripper(0)

    post_handover_pose_agent1 = handover_pose_agent1.copy()
    post_handover_pose_agent1[2] += 0.08
    if not _move(planner, post_handover_pose_agent1, move_id=1):
        planner.close()
        return -1

    # Segment 3: agent 0 leaves the handover area gradually and moves toward
    # agent 1's initial perception pose, reducing abrupt IK/planning changes
    # near the cube while agent 1 is holding it.
    planner.set_mode_label("perception")
    agent0_retreat_pose = pre_handover_pose_agent0.copy()
    agent0_high_retreat_pose = agent0_retreat_pose.copy()
    agent0_high_retreat_pose[2] += 0.08
    agent0_mid_pose = _position_mid_pose(agent0_high_retreat_pose, AGENT1_PERCEPTION_POSE)
    if not _move_pose_sequence(
        planner,
        [
            agent0_retreat_pose,
            agent0_high_retreat_pose,
            agent0_mid_pose,
            AGENT1_PERCEPTION_POSE,
        ],
        move_id=0,
        mode_label="perception",
    ):
        planner.close()
        return -1

    place_pose = _side_grasp_pose(
        env,
        agent_id=1,
        cube_center=env.right_goal_region.pose.p[0].cpu().numpy()
        + np.array([0.0, 0.0, env.cube_half_size_value]),
        side="positive_y",
    )
    pre_place_pose = _pre_side_grasp_pose(place_pose, side="positive_y")
    if not _move(planner, pre_place_pose, move_id=1):
        planner.close()
        return -1
    if not _move(planner, place_pose, move_id=1):
        planner.close()
        return -1

    res = planner.open_gripper(1)
    planner.close()
    return res
