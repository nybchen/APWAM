import numpy as np
import sapien

from robofactory.tasks import PickMeatFromPotEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


def solve(env: PickMeatFromPotEnv, seed=None, debug=False, vis=False):
    """
    Motion planning solution for picking meat from pot with active perception.
    
    Two robots coordination:
    - Robot 0 (perception): Moves to observe the pot (active perception)
    - Robot 1 (picker): Picks the meat and places it in the goal region
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
    
    # === Phase 1: Position perception robot to observe the pot ===
    # Robot 0 moves to observe the pot
    #Pose([-0.00315437, -0.286263, 0.457501], [-0.251162, 0.661668, 0.654008, 0.267184])
    planner.set_mode_label("perception")
    perception_pose = np.array([-0.00315437, -0.286263, 0.457501, -0.251162, 0.661668, 0.654008, 0.267184])  # Looking down pose
    planner.move_to_pose_with_screw(perception_pose, move_id=0, mode_label="perception")
    
    # === Phase 2: Robot 1 picks meat and places it in goal region ===
    # Get grasp pose for meat
    planner.set_mode_label("perception")
    meat_grasp_pose = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=0)
    meat_grasp_pose[2] += 0.05  # Approach from above
    planner.move_to_pose_with_screw(meat_grasp_pose, move_id=1)
    
    # Move down to grasp
    meat_grasp_pose[2] -= 0.05
    planner.move_to_pose_with_screw(meat_grasp_pose, move_id=1)
    planner.close_gripper(close_id=[1])
    
    # Lift meat
    meat_grasp_pose[2] += 0.3
    planner.move_to_pose_with_screw(meat_grasp_pose, move_id=1)
    
    # Move to goal region (above goal) target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    target_pose = planner.get_grasp_pose_for_stack(meat_grasp_pose, env.goal_region)
    planner.move_to_pose_with_screw(target_pose, move_id=1)
    
    
    # Release meat
    res = planner.open_gripper(open_id=[1])
    
    
    return res
