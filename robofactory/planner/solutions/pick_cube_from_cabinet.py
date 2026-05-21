import numpy as np
import sapien

from robofactory.tasks import PickCubeFromCabinetEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


def solve(env: PickCubeFromCabinetEnv, seed=None, debug=False, vis=False):
    """
    Motion planning solution for picking cube from cabinet with active perception.
    
    Three robots coordination:
    - Robot 0 (perception): Moves to observe the scene from top
    - Robot 1 (drawer opener): Opens the cabinet drawer by pulling the handle
    - Robot 2 (picker): Picks the cube from the drawer and places it in the goal region
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
    
    # === Phase 1: Robot 2 positions for top-down observation ===
    # Move perception robot to observe the whole scene from top
    # Pose([0, 0, 0.6], [0.00106633, 0.998201, -0.0134043, 0.0584223])
    #Pose([-0.286745, 0, 0.489328], [0.00106629, 0.998201, -0.0134043, 0.0584223])
    #Pose([-0.52402, 0.409508, 0.344259], [0.243583, 0.73186, -0.528351, 0.354816])
    perception_pose = np.array([-0.52402, 0.409508, 0.344259, 0.243583, 0.73186, -0.528351, 0.354816])  # Top-down view
    planner.move_to_pose_with_screw(perception_pose, move_id=2, mode_label="perception")
    # Robot 1 opens the cabinet
    grasp_pose = planner.get_grasp_pose_w_labeled_direction(actor=env.cabinet, actor_data=env.annotation_data['cabinet'], pre_dis=0, id=0)
    grasp_pose[0] -= 0.05
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    grasp_pose[0] += 0.05
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    planner.close_gripper(close_id=[1])
    grasp_pose[0] -= 0.2
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    #   Pose([-0.745552, -0.462443, 0.346143], [-0.168495, 0.838735, 0.33362, 0.396019])
    robot1_pose= np.array([-0.745552, -0.462443, 0.346143, -0.168495, 0.838735, 0.33362, 0.396019])
    planner.move_to_pose_with_screw(robot1_pose, move_id=1)

    # Robot 0 Picks the cube
    grasp_pose = planner.get_grasp_pose_from_obb(env.cube)
    grasp_pose[2] += 0.04
    planner.move_to_pose_with_screw(grasp_pose, move_id=0)
    grasp_pose[2] -= 0.04
    planner.move_to_pose_with_screw(grasp_pose, move_id=0)
    planner.close_gripper(close_id=[0])
    target_pose = planner.get_grasp_pose_for_stack(grasp_pose, env.goal_region)
    planner.move_to_pose_with_screw(target_pose, move_id=0)
    planner.open_gripper(open_id=[0])
    res = planner.close()
    return res
