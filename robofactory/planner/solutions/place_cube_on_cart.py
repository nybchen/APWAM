import numpy as np
import sapien

from robofactory.tasks import PlaceCubeOnCartEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


def solve(env: PlaceCubeOnCartEnv, seed=None, debug=False, vis=False):
    """
    Motion planning solution for placing a cube from the table onto the cart with active perception.

    Two robots coordination:
    - Robot 0 (perception): Moves to observe the scene from top
    - Robot 1 (picker): Picks the cube from the table and places it on the cart goal region
    """
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped

    # Phase 1: Move perception robot to a good top-down view
    # (these poses are heuristic and may need tuning)
    #Pose([-0.56162, -0.26344, 0.477174], [-0.203068, 0.801166, 0.27616, 0.490543])
    planner.set_mode_label("perception")
    perception_pose = np.array([-0.56162, -0.26344, 0.477174, -0.203068, 0.801166, 0.27616, 0.490543])
    planner.move_to_pose_with_screw(perception_pose, move_id=0, mode_label="perception")

    planner.set_mode_label("perception")
    grasp_pose = planner.get_grasp_pose_from_obb(env.cube, 0)
    grasp_pose[2] += 0.04
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    grasp_pose[2] -= 0.04
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    planner.close_gripper(close_id=[1])
    grasp_pose[2] += 0.2
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    
    place_pose = np.array([0.013364, -0.237383, 0.34313, -0.497254, 0.451086, 0.562979, 0.48199])
    planner.move_to_pose_with_screw(place_pose, move_id=1)
    place_pose = np.array([-0.0266592, 0.0798912, 0.330056, -0.497254, 0.451085, 0.56298, 0.48199])
    planner.move_to_pose_with_screw(place_pose, move_id=1)
    
    res = planner.open_gripper(open_id=[1])
    return res
