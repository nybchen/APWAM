import numpy as np
import sapien

from robofactory.tasks import PickMeatFromMicrowaveEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


def solve(env: PickMeatFromMicrowaveEnv, seed=None, debug=False, vis=False):
    """
    Motion planning solution for picking meat from microwave with active perception.
    
    Three robots coordination:
    - Robot 0 (perception): Moves to observe the scene from top
    - Robot 1 (door opener): Opens the microwave door by pulling the handle
    - Robot 2 (picker): Picks the meat from the microwave and places it in the goal region
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
    

    robot1_pose= np.array([-0.745552, -0.462443, 0.546143, -0.168495, 0.838735, 0.33362, 0.396019])
    planner.move_to_pose_with_screw(robot1_pose, move_id=1)
    
    # PPose([Pose([-0.711235, 0.21192, 0.350585], [0.0214251, 0.980529, -0.0695068, 0.182406])
    robot0_pose= np.array([-0.711235, 0.21192, 0.350585, 0.0214251, 0.980529, -0.0695068, 0.182406])
    planner.move_to_pose_with_screw(robot0_pose, move_id=0)
    
    # === Phase 2: Robot 2 opens the microwave door ===
    # Get grasp pose for the door handle (id=0 in microwave_annotated/models.py)
    door_handle_grasp_pose = planner.get_grasp_pose_w_labeled_direction(
        actor=env.microwave, 
        actor_data=env.annotation_data['microwave'], 
        pre_dis=0, 
        id=0  # First contact point for door handle
    )
    
    # Approach handle from front
    door_handle_grasp_pose[0] -= 0.05
    door_handle_grasp_pose[2] += 0.02
    planner.move_to_pose_with_screw(door_handle_grasp_pose, move_id=2)
    
    door_handle_grasp_pose[0] += 0.07
    planner.move_to_pose_with_screw(door_handle_grasp_pose, move_id=2)

    # Close gripper to grasp handle
    planner.close_gripper(close_id=[2])
    
    # Move to open the microwave door
    #Pose([-0.36139, -0.05641, 0.219849], [0.263025, 0.656391, -0.263059, 0.65633])
    pose = np.array([-0.36139, -0.05641, 0.219849, 0.263025, 0.656391, -0.263059, 0.65633])
    planner.move_to_pose_with_screw(pose, move_id=2)
    
    # Pose([-0.462075, 0.162773, 0.199797], [0.514068, 0.485509, -0.514162, 0.48544])
    pose = np.array([-0.462075, 0.162773, 0.219797, 0.514068, 0.485509, -0.514162, 0.48544])
    planner.move_to_pose_with_screw(pose, move_id=2)
    planner.open_gripper(open_id=[2])
    
    
    perception_pose = np.array([-0.52402, 0.409508, 0.544259, 0.243583, 0.73186, -0.528351, 0.354816])  # Top-down view
    planner.move_to_pose_with_screw(perception_pose, move_id=2)
      
    
    # Robot 1 picks the meat
    
    
    grasp_pose = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=2)
    grasp_pose[0] -= 0.1
    grasp_pose[2] += 0.1
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    grasp_pose[2] -= 0.1
    grasp_pose[0] += 0.11
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    planner.close_gripper(close_id=[1])
    target_pose = planner.get_grasp_pose_for_stack(grasp_pose, env.goal_region)
    target_pose[2] += 0.1
    target_pose[0] -= 0.05
    planner.move_to_pose_with_screw(target_pose, move_id=1)
    res = planner.open_gripper(open_id=[1])
    
    return res

