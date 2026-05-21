"""Two-robot active-perception handover solver.

The task supports both left-to-right and right-to-left directions. The source
arm picks the cube from its side, moves it to a central handover pose, the
target arm grasps it, and the target arm places it into the opposite goal
region. The non-manipulating arm is moved to a wrist-camera viewing pose when
possible, so the recorded trajectory contains perception/action mode switches.
"""

import numpy as np

from robofactory.tasks import TwoRobotsHandoverActiveEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


OBSERVER_POSE = {
    0: np.array(
        [
            -0.930056,
            -0.419985,
            0.47093,
            0.990189,
            -0.00493824,
            0.0362513,
            0.134862,
        ]
    ),
    1: np.array(
        [
            0.930056,
            0.419985,
            0.47093,
            -0.134862,
            -0.0362513,
            -0.00493824,
            0.990189,
        ]
    ),
}


def _lifted_grasp_pose(planner, actor, agent_id, lift=0.04):
    pose = planner.get_grasp_pose_from_obb(actor, agent_id)
    above_pose = pose.copy()
    above_pose[2] += lift
    return pose, above_pose


def _move_or_fail(planner, pose, move_id, mode_label="action"):
    res = planner.move_to_pose_with_screw(pose, move_id=move_id, mode_label=mode_label)
    if res == -1:
        return False
    return True


def solve_handover(
    env: TwoRobotsHandoverActiveEnv,
    seed=101,
    debug=False,
    vis=False,
    source_id=None,
    target_id=None,
):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True,
    )
    env = env.unwrapped

    if source_id is None:
        source_id = env.source_agent_id
    if target_id is None:
        target_id = env.target_agent_id
    if debug:
        print(
            f"[handover-active] direction={env.handover_direction} "
            f"source=agent{source_id} target=agent{target_id}"
        )

    # Phase 1: target arm observes while source arm picks up the cube.
    planner.set_mode_label("perception")
    if not _move_or_fail(
        planner, OBSERVER_POSE[target_id], move_id=target_id, mode_label="perception"
    ):
        planner.close()
        return -1

    planner.set_mode_label("perception")
    grasp_pose, above_grasp_pose = _lifted_grasp_pose(planner, env.cube, source_id)
    if not _move_or_fail(planner, above_grasp_pose, move_id=source_id):
        planner.close()
        return -1
    if not _move_or_fail(planner, grasp_pose, move_id=source_id):
        planner.close()
        return -1
    planner.close_gripper(source_id)

    source_lift_pose = grasp_pose.copy()
    source_lift_pose[2] += 0.14
    if not _move_or_fail(planner, source_lift_pose, move_id=source_id):
        planner.close()
        return -1

    # Phase 2: source brings the cube to the handover pose; target grasps it.
    handover_pose = source_lift_pose.copy()
    handover_pose[:3] = np.array([0.0, 0.0, 0.18])
    if not _move_or_fail(planner, handover_pose, move_id=source_id):
        planner.close()
        return -1

    target_grasp_pose, target_above_pose = _lifted_grasp_pose(
        planner, env.cube, target_id, lift=0.035
    )
    if not _move_or_fail(planner, target_above_pose, move_id=target_id):
        planner.close()
        return -1
    if not _move_or_fail(planner, target_grasp_pose, move_id=target_id):
        planner.close()
        return -1

    planner.set_mode_label("action", agent_ids=[source_id, target_id])
    planner.close_gripper(target_id)
    planner.open_gripper(source_id)

    target_lift_pose = target_grasp_pose.copy()
    target_lift_pose[2] += 0.12
    if not _move_or_fail(planner, target_lift_pose, move_id=target_id):
        planner.close()
        return -1

    # Phase 3: source switches back to perception while target places the cube.
    planner.set_mode_label("perception")
    if not _move_or_fail(
        planner, OBSERVER_POSE[source_id], move_id=source_id, mode_label="perception"
    ):
        planner.close()
        return -1

    place_pose = planner.get_grasp_pose_for_stack(
        target_lift_pose, env.target_goal_region, height_offset=0.05
    )
    pre_place_pose = place_pose.copy()
    pre_place_pose[2] += 0.08
    if not _move_or_fail(planner, pre_place_pose, move_id=target_id):
        planner.close()
        return -1
    if not _move_or_fail(planner, place_pose, move_id=target_id):
        planner.close()
        return -1
    res = planner.open_gripper(target_id)
    planner.close()
    return res


def solve(env: TwoRobotsHandoverActiveEnv, seed=101, debug=False, vis=False):
    return solve_handover(env, seed=seed, debug=debug, vis=vis)
