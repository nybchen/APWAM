import numpy as np

from robofactory.tasks import TwoRobotsHandoverActiveBEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


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

HANDOVER_CENTER = np.array([0.0, 0.0, 0.26])
PRE_APPROACH_DIST = 0.14
HANDOVER_RETREAT_DIST = 0.045
FINGER_DEPTH = 0.025


def solve(env: TwoRobotsHandoverActiveBEnv, seed=None, debug=False, vis=False, reset=True):
    """Right-to-left handover: agent 1 gives the cube to agent 0."""
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
        AGENT0_PERCEPTION_POSE, move_id=0, mode_label="perception"
    ) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: agent0 initial perception")
        planner.close()
        return -1

    pick_pose = planner.get_grasp_pose_from_obb(env.cube, 1)
    w, x, y, z = pick_pose[3:]
    c = np.sqrt(0.5)
    s = np.sqrt(0.5)
    pick_pose[3:] = np.array(
        [
            w * c - z * s,
            x * c + y * s,
            y * c - x * s,
            w * s + z * c,
        ]
    )
    pre_pick_pose = pick_pose.copy()
    pre_pick_pose[2] += 0.05
    if planner.move_to_pose_with_screw(pre_pick_pose, move_id=1) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: agent1 pre-pick")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(pick_pose, move_id=1) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: agent1 pick")
        planner.close()
        return -1
    planner.close_gripper(1)

    approaching = np.array([0.0, -1.0, 0.0])
    closing = np.array([-1.0, 0.0, 0.0])
    tcp_center = HANDOVER_CENTER + approaching * handover_depth
    pose = env.agent.agents[1].build_grasp_pose(approaching, closing, tcp_center)
    agent1_handover_pose = np.array(list(pose.p) + list(pose.q))
    agent1_pre_handover_pose = agent1_handover_pose.copy()
    agent1_pre_handover_pose[1] += PRE_APPROACH_DIST

    agent0_ready_mid_pose = AGENT0_HANDOVER_READY_POSE.copy()
    agent0_ready_mid_pose[:3] = 0.5 * (
        AGENT0_PERCEPTION_POSE[:3] + AGENT0_HANDOVER_READY_POSE[:3]
    )
    if planner.move_to_pose_with_screw(
        [agent0_ready_mid_pose, agent1_pre_handover_pose],
        move_id=[0, 1],
        mode_label=["perception", "action"],
    ) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: pre-handover")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(
        [AGENT0_HANDOVER_READY_POSE, agent1_handover_pose],
        move_id=[0, 1],
        mode_label=["perception", "action"],
    ) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: enter handover")
        planner.close()
        return -1

    approaching = np.array([0.0, 1.0, 0.0])
    closing = np.array([0.0, 0.0, -1.0])
    tcp_center = HANDOVER_CENTER + approaching * handover_depth
    pose = env.agent.agents[0].build_grasp_pose(approaching, closing, tcp_center)
    agent0_handover_pose = np.array(list(pose.p) + list(pose.q))
    agent0_pre_handover_pose = agent0_handover_pose.copy()
    agent0_pre_handover_pose[1] -= PRE_APPROACH_DIST

    if planner.move_to_pose_with_screw(agent0_pre_handover_pose, move_id=0) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: agent0 pre-handover")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(agent0_handover_pose, move_id=0) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: agent0 handover grasp")
        planner.close()
        return -1

    planner.set_mode_label("action", agent_ids=[0, 1])
    planner.close_gripper(0)
    planner.open_gripper(1)

    agent0_retreat_pose = agent0_handover_pose.copy()
    agent0_retreat_pose[1] -= HANDOVER_RETREAT_DIST
    agent1_retreat_pose = agent1_handover_pose.copy()
    agent1_retreat_pose[1] += HANDOVER_RETREAT_DIST
    if planner.move_to_pose_with_screw(
        [agent0_retreat_pose, agent1_retreat_pose],
        move_id=[0, 1],
        mode_label=["action", "perception"],
    ) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: retreat after handover")
        planner.close()
        return -1

    pre_drop_pose = AGENT0_DROP_POSE.copy()
    pre_drop_pose[2] += 0.12
    if planner.move_to_pose_with_screw(
        [pre_drop_pose, AGENT1_POST_HANDOVER_POSE],
        move_id=[0, 1],
        mode_label=["action", "perception"],
    ) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: carry to pre-drop")
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(
        [AGENT0_DROP_POSE, AGENT1_POST_HANDOVER_POSE],
        move_id=[0, 1],
        mode_label=["action", "perception"],
    ) == -1:
        print("[TwoRobotsHandoverActiveB] failed motion: drop")
        planner.close()
        return -1

    res = planner.open_gripper(0)
    planner.close()
    return res
