import numpy as np

from robofactory.tasks import TwoRobotsHandoverActiveAEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


AGENT0_PERCEPTION_POSE = np.array(
    [-0.00119789, 0.0493949, 0.143102, 0.41325, -0.578902, -0.561516, -0.422844]
)
AGENT1_PERCEPTION_POSE = np.array(
    [0.217725, -0.0798748, 0.245282, 0.528184, 0.359683, -0.721849, 0.265677]
)
AGENT1_HANDOVER_READY_POSE = np.array(
    [0.12493, 0.0787286, 0.346096, 0.528184, 0.359683, -0.721849, 0.265677]
)
AGENT0_POST_HANDOVER_POSE = np.array(
    [0.0, 0.039336, 0.26, -0.36653, 0.605357, 0.604461, 0.365823]
)
AGENT1_DROP_POSE = np.array(
    [0.0, 0.285, 0.151, 0.0958634, 0.733145, -0.67211, 0.0397062]
)
HANDOVER_CENTER = np.array([0.0, 0.0, 0.26])
PRE_APPROACH_DIST = 0.14
HANDOVER_RETREAT_DIST = 0.045


def _pre_top_grasp_pose(grasp_pose, distance=0.05):
    pre_pose = grasp_pose.copy()
    pre_pose[2] += distance
    return pre_pose


def _side_grasp_pose(env, agent_id: int, cube_center, side: str):
    if side == "negative_y":
        approaching = np.array([0.0, 1.0, 0.0])
        closing = np.array([1.0, 0.0, 0.0])
    elif side == "positive_y":
        approaching = np.array([0.0, -1.0, 0.0])
        closing = np.array([-1.0, 0.0, 0.0])
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


def _move(planner, pose, move_id, mode_label="action", name=None):
    res = planner.move_to_pose_with_screw(pose, move_id=move_id, mode_label=mode_label)
    if res == -1 and name is not None:
        print(f"[TwoRobotsHandoverActiveA] failed motion: {name}")
    return res != -1


def _move_pose_sequence(planner, poses, move_id, mode_label="action", names=None):
    names = names or [None] * len(poses)
    for pose, name in zip(poses, names):
        if not _move(planner, pose, move_id=move_id, mode_label=mode_label, name=name):
            return False
    return True


def _move_joint_pose_sequence(planner, pose_pairs, mode_label, names=None):
    names = names or [None] * len(pose_pairs)
    for (pose0, pose1), name in zip(pose_pairs, names):
        res = planner.move_to_pose_with_screw(
            [pose0, pose1], move_id=[0, 1], mode_label=mode_label
        )
        if res == -1:
            if name is not None:
                print(f"[TwoRobotsHandoverActiveA] failed joint motion: {name}")
            return False
    return True


def _position_mid_pose(start_pose, target_pose):
    mid_pose = target_pose.copy()
    mid_pose[:3] = 0.5 * (start_pose[:3] + target_pose[:3])
    return mid_pose


def _y_offset_pose(pose, delta_y):
    offset_pose = pose.copy()
    offset_pose[1] += delta_y
    return offset_pose


def solve(env: TwoRobotsHandoverActiveAEnv, seed=None, debug=False, vis=False, reset=True):
    """
    Left-to-right handover with explicit opposing gripper geometry.

    Agent 0 picks the cube from above, then rotates into a side-facing
    handover pose and holds the cube's side faces. Agent 1 approaches from
    the opposite side with vertical closing, so it clamps the cube from top
    and bottom during transfer.
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

    # Segment 1: agent 1 watches while agent 0 picks from the left goal region.
    planner.set_mode_label("perception")
    if not _move(
        planner,
        AGENT1_PERCEPTION_POSE,
        move_id=1,
        mode_label="perception",
        name="agent1 initial perception pose",
    ):
        planner.close()
        return -1

    pick_pose = planner.get_grasp_pose_from_obb(env.cube, 0)
    pre_pick_pose = _pre_top_grasp_pose(pick_pose)

    planner.set_mode_label("perception")
    if not _move(planner, pre_pick_pose, move_id=0, name="agent0 pre-pick"):
        planner.close()
        return -1
    if not _move(planner, pick_pose, move_id=0, name="agent0 pick"):
        planner.close()
        return -1
    planner.close_gripper(0)

    handover_pose_agent0 = _side_grasp_pose(
        env, agent_id=0, cube_center=HANDOVER_CENTER, side="negative_y"
    )
    pre_handover_pose_agent0 = _pre_side_grasp_pose(
        handover_pose_agent0, side="negative_y"
    )
    agent1_ready_mid_pose = _position_mid_pose(
        AGENT1_PERCEPTION_POSE, AGENT1_HANDOVER_READY_POSE
    )
    if not _move_joint_pose_sequence(
        planner,
        [
            (pre_handover_pose_agent0, agent1_ready_mid_pose),
            (handover_pose_agent0, AGENT1_HANDOVER_READY_POSE),
        ],
        mode_label=["action", "perception"],
        names=[
            "agent0 lift to pre-handover while agent1 raises",
            "agent0 enter handover while agent1 reaches ready pose",
        ],
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
    if not _move_pose_sequence(
        planner,
        [
            pre_handover_pose_agent1,
            handover_pose_agent1,
        ],
        move_id=1,
        names=["agent1 pre-handover approach", "agent1 top-bottom handover grasp"],
    ):
        planner.close()
        return -1

    planner.set_mode_label("action", agent_ids=[0, 1])
    planner.close_gripper(1)
    planner.open_gripper(0)

    pre_drop_pose = _pre_top_grasp_pose(AGENT1_DROP_POSE, distance=0.12)
    retreat_pose_agent0 = _y_offset_pose(handover_pose_agent0, -HANDOVER_RETREAT_DIST)
    retreat_pose_agent1 = _y_offset_pose(handover_pose_agent1, HANDOVER_RETREAT_DIST)

    # Segment 3: both arms first separate from the handover contact, then
    # agent 0 goes to its post-handover observation pose while agent 1 drops.
    if not _move_joint_pose_sequence(
        planner,
        [
            (retreat_pose_agent0, retreat_pose_agent1),
            (AGENT0_POST_HANDOVER_POSE, pre_drop_pose),
        ],
        mode_label=["perception", "action"],
        names=[
            "both agents retreat after handover",
            "agent0 post-handover observation while agent1 carries to pre-drop",
        ],
    ):
        planner.close()
        return -1

    if not _move_joint_pose_sequence(
        planner,
        [
            (AGENT0_POST_HANDOVER_POSE, AGENT1_DROP_POSE),
        ],
        mode_label=["perception", "action"],
        names=[
            "agent1 drops cube while agent0 watches",
        ],
    ):
        planner.close()
        return -1
    res = planner.open_gripper(1)

    planner.close()
    return res
