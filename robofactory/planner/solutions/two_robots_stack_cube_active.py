"""TwoRobotsStackCubeActive solver with dynamic role assignment.

Whichever arm is closer to the cube centroid takes the manipulator role
(picks cubeA -> goal, then stacks cubeB on cubeA). The other arm parks
at a viewing pose for active perception and stays there for the rest
of the episode.

Observer EEF poses are world-frame [xyz, wxyz]. The agent-1 pose was the
original hardcoded baseline. The agent-0 pose is its mirror under a
180-degree rotation about z, since the two arm bases sit at y = -0.6 and
y = +0.6 facing opposite directions.
"""

import numpy as np

from robofactory.tasks import TwoRobotsStackCubeActiveEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


OBSERVER_POSE = {
    1: np.array(
        [0.371961, 0.149633, 0.312432,
         -0.445288, -0.243162, 0.837862, -0.201438]
    ),
    0: np.array(
        [-0.371961, -0.149633, 0.312432,
         0.201438, -0.837862, -0.243162, -0.445288]
    ),
}

ARM_BASE_XY = {
    0: np.array([0.0, -0.6]),
    1: np.array([0.0, 0.6]),
}


def _pick_manipulator(env) -> tuple[int, int]:
    cubeA_xy = env.cubeA.pose.p[0, :2].cpu().numpy()
    cubeB_xy = env.cubeB.pose.p[0, :2].cpu().numpy()
    centroid = 0.5 * (cubeA_xy + cubeB_xy)
    d0 = np.linalg.norm(centroid - ARM_BASE_XY[0])
    d1 = np.linalg.norm(centroid - ARM_BASE_XY[1])
    return (0, 1) if d0 <= d1 else (1, 0)


def solve(env: TwoRobotsStackCubeActiveEnv, seed=101, debug=False, vis=False):
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

    manip_id, obs_id = _pick_manipulator(env)
    if debug:
        print(f"[stack-active] manipulator=agent{manip_id} observer=agent{obs_id}")

    # Park the observer at its viewing pose.
    planner.set_mode_label("perception")
    planner.move_to_pose_with_screw(OBSERVER_POSE[obs_id], move_id=obs_id, mode_label="perception")

    # Pick cubeA, place at goal region.
    planner.set_mode_label("perception")
    grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, manip_id)
    grasp_poseA[2] += 0.04
    planner.move_to_pose_with_screw(grasp_poseA, move_id=manip_id)
    grasp_poseA[2] -= 0.04
    planner.move_to_pose_with_screw(grasp_poseA, move_id=manip_id)
    planner.close_gripper(manip_id)

    target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    planner.move_to_pose_with_screw(target_poseA, move_id=manip_id)
    planner.open_gripper(manip_id)

    target_poseA[2] += 0.10
    planner.move_to_pose_with_screw(target_poseA, move_id=manip_id)

    # Pick cubeB, stack on cubeA.
    grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, manip_id)
    grasp_poseB[2] += 0.04
    planner.move_to_pose_with_screw(grasp_poseB, move_id=manip_id)
    grasp_poseB[2] -= 0.04
    planner.move_to_pose_with_screw(grasp_poseB, move_id=manip_id)
    planner.close_gripper(manip_id)
    grasp_poseB[2] += 0.10
    planner.move_to_pose_with_screw(grasp_poseB, move_id=manip_id)

    stack_poseB = planner.get_grasp_pose_for_stack(grasp_poseB, env.cubeA)
    planner.move_to_pose_with_screw(stack_poseB, move_id=manip_id)
    res = planner.open_gripper(manip_id)
    planner.close()
    return res
