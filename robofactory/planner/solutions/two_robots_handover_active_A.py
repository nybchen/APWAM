import numpy as np

from robofactory.tasks import TwoRobotsHandoverActiveAEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


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
FINGER_DEPTH = 0.025


def solve(env: TwoRobotsHandoverActiveAEnv, seed=None, debug=False, vis=False, reset=True):
    """Left-to-right handover: agent 0 gives the cube to agent 1."""
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
    half_size = float(env.cube_half_size_value)
    handover_depth = -half_size + min(FINGER_DEPTH, half_size)

    planner.set_mode_label("perception")
    if planner.move_to_pose_with_screw(
        AGENT1_PERCEPTION_POSE, move_id=1, mode_label="perception"
    ) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: agent1 initial perception")
        planner.close()
        return -1

    pick_pose = planner.get_grasp_pose_from_obb(env.cube, 0)
    pre_pick_pose = pick_pose.copy()
    pre_pick_pose[2] += 0.05
    if planner.move_to_pose_with_screw(pre_pick_pose, move_id=0) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: agent0 pre-pick")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(pick_pose, move_id=0) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: agent0 pick")
        planner.close()
        return -1
    planner.close_gripper(0)

    approaching = np.array([0.0, 1.0, 0.0])
    closing = np.array([1.0, 0.0, 0.0])
    tcp_center = HANDOVER_CENTER + approaching * handover_depth
    pose = env.agent.agents[0].build_grasp_pose(approaching, closing, tcp_center)
    agent0_handover_pose = np.array(list(pose.p) + list(pose.q))
    agent0_pre_handover_pose = agent0_handover_pose.copy()
    agent0_pre_handover_pose[1] -= PRE_APPROACH_DIST

    agent1_ready_mid_pose = AGENT1_HANDOVER_READY_POSE.copy()
    agent1_ready_mid_pose[:3] = 0.5 * (
        AGENT1_PERCEPTION_POSE[:3] + AGENT1_HANDOVER_READY_POSE[:3]
    )
    if planner.move_to_pose_with_screw(
        [agent0_pre_handover_pose, agent1_ready_mid_pose],
        move_id=[0, 1],
        mode_label=["action", "perception"],
    ) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: pre-handover")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(
        [agent0_handover_pose, AGENT1_HANDOVER_READY_POSE],
        move_id=[0, 1],
        mode_label=["action", "perception"],
    ) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: enter handover")
        planner.close()
        return -1

    approaching = np.array([0.0, -1.0, 0.0])
    closing = np.array([0.0, 0.0, 1.0])
    tcp_center = HANDOVER_CENTER + approaching * handover_depth
    pose = env.agent.agents[1].build_grasp_pose(approaching, closing, tcp_center)
    agent1_handover_pose = np.array(list(pose.p) + list(pose.q))
    agent1_pre_handover_pose = agent1_handover_pose.copy()
    agent1_pre_handover_pose[1] += PRE_APPROACH_DIST

    if planner.move_to_pose_with_screw(agent1_pre_handover_pose, move_id=1) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: agent1 pre-handover")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(agent1_handover_pose, move_id=1) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: agent1 handover grasp")
        planner.close()
        return -1

    planner.set_mode_label("action", agent_ids=[0, 1])
    planner.close_gripper(1)
    planner.open_gripper(0)

    agent0_retreat_pose = agent0_handover_pose.copy()
    agent0_retreat_pose[1] -= HANDOVER_RETREAT_DIST
    agent1_retreat_pose = agent1_handover_pose.copy()
    agent1_retreat_pose[1] += HANDOVER_RETREAT_DIST
    if planner.move_to_pose_with_screw(
        [agent0_retreat_pose, agent1_retreat_pose],
        move_id=[0, 1],
        mode_label=["perception", "action"],
    ) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: retreat after handover")
        planner.close()
        return -1

    pre_drop_pose = AGENT1_DROP_POSE.copy()
    pre_drop_pose[2] += 0.12
    if planner.move_to_pose_with_screw(
        [AGENT0_POST_HANDOVER_POSE, pre_drop_pose],
        move_id=[0, 1],
        mode_label=["perception", "action"],
    ) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: carry to pre-drop")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(
        [AGENT0_POST_HANDOVER_POSE, AGENT1_DROP_POSE],
        move_id=[0, 1],
        mode_label=["perception", "action"],
    ) == -1:
        print("[TwoRobotsHandoverActiveA] failed motion: drop")
        planner.close()
        return -1

    res = planner.open_gripper(1)
    planner.close()
    return res
