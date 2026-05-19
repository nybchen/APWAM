import numpy as np
import sapien

from robofactory.tasks import TwoRobotsStackCubeActiveEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: TwoRobotsStackCubeActiveEnv, seed=101, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped
    # env.debug_robot_sensors()
    # Task decomposition
    # Step 1: Two robots pick up the cubes
    # Calculate if the cubes are closer to agent0 or agent1
    # print(f"cubeA_pose: {env.cubeA.pose.p}")
    # print(f"cubeB_pose: {env.cubeB.pose.p}")
    # cubes_y = (env.cubeA.pose.p[0][1] + env.cubeB.pose.p[0][1]) / 2
    # # print(f"cubes_y: {cubes_y}")
    # # Agent 1 as camera agenttarget_joint_pos
    # # Agent 0 pick up cube 1 and move it to the goal region
    # if cubes_y < 0:
    # # agent1_target_joint_pos = np.array([-0.90851676, -0.6334094, 1.5198542, -2.4929106, -0.27709103, 2.6039643, 2.405765])
    #     agent1_target_joint_pos = np.array([0.371961, 0.149633, 0.312432, -0.445288, -0.243162, 0.837862, -0.201438])
    #     # key is "panda_wristcam-0"
    #     # - env.step是maniskill包装好的api，通过它传递action进去，并获取观察，奖励等信息。
    #     #   - 单智能体环境下，传递一个action列表即可，shape为单智能体所需的action的shape
    #     # #   - 多智能体环境下，传递一个字典，key为智能体名，默认命名为panda-0，panda-1...\
    #     # agent0_current_joint_pos = env.agent.agents[0].robot.get_qpos()[0, :-1].cpu().numpy()
    #     # env.step({"panda_wristcam-0": agent0_current_joint_pos, "panda_wristcam-1": agent1_target_joint_pos})
    #     planner.move_to_pose_with_screw(agent1_target_joint_pos, move_id=1)
    #     grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 0)
    #     grasp_poseA[2] += 0.04
    #     print(f"grasp_poseA: {grasp_poseA}")
        
    #     planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
    #     grasp_poseA[2] -= 0.04
    #     planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
    #     planner.close_gripper(0)
    #     target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    #     planner.move_to_pose_with_screw(target_poseA, move_id=0)
    #     planner.open_gripper(0)

    #     target_poseA[2] += 0.10
    #     planner.move_to_pose_with_screw(target_poseA, move_id=0)


    #     grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 0)
    #     grasp_poseB[2] += 0.04
    #     planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
    #     grasp_poseB[2] -= 0.04
    #     planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
    #     planner.close_gripper(0)
    #     grasp_poseB[2] += 0.10
    #     target_poseA = planner.get_grasp_pose_for_stack(grasp_poseB, env.cubeA)
    #     planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
    #     planner.move_to_pose_with_screw(target_poseA, move_id=0)
    #     planner.open_gripper(0)
    # else:
    #     # make agent 0 as camera agent, inverse the logic
    #     # agent0_target_joint_pos = np.array([0.371961, 0.149633, 0.312432, -0.445288, -0.243162, 0.837862, -0.201438])Pose([-0.355946, -0.393805, 0.313839], [-0.24827, 0.70622, 0.329728, 0.575235]) Pose([-0.2991, -0.302628, 0.313777], [-0.251538, 0.808955, 0.241039, 0.473519])
    # # Pose([-0.223014, -0.216813, 0.260888], [-0.205299, 0.792666, 0.271186, 0.505956])
    #     agent0_target_joint_pos = np.array([-0.223014, -0.216813, 0.260888,-0.205299, 0.792666, 0.271186, 0.505956])
    #     planner.move_to_pose_with_screw(agent0_target_joint_pos, move_id=0)
    #     grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 1)
    #     grasp_poseA[2] += 0.04
    #     # print(f"grasp_poseA: {grasp_poseA}")
    #     planner.move_to_pose_with_screw(grasp_poseA, move_id=1)
    #     grasp_poseA[2] -= 0.04
    #     planner.move_to_pose_with_screw(grasp_poseA, move_id=1)
    #     planner.close_gripper(1)
    #     target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    #     planner.move_to_pose_with_screw(target_poseA, move_id=1)
    #     planner.open_gripper(1)
    #     target_poseA[2] += 0.10
    #     planner.move_to_pose_with_screw(target_poseA, move_id=1)
    #     grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 1)
    #     grasp_poseB[2] += 0.04
    #     planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
    #     grasp_poseB[2] -= 0.04
    #     planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
    #     planner.close_gripper(1)
    #     grasp_poseB[2] += 0.10
    #     planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
    #     target_poseA = planner.get_grasp_pose_for_stack(grasp_poseB, env.cubeA)
    #     planner.move_to_pose_with_screw(target_poseA, move_id=1)
    #     planner.open_gripper(1)
    agent1_target_joint_pos = np.array([-0.90851676, -0.6334094, 1.5198542, -2.4929106, -0.27709103, 2.6039643, 2.405765])
    agent1_target_joint_pos = np.array([0.371961, 0.149633, 0.312432, -0.445288, -0.243162, 0.837862, -0.201438])
    # key is "panda_wristcam-0"
    # - env.step是maniskill包装好的api，通过它传递action进去，并获取观察，奖励等信息。
    #   - 单智能体环境下，传递一个action列表即可，shape为单智能体所需的action的shape
    # #   - 多智能体环境下，传递一个字典，key为智能体名，默认命名为panda-0，panda-1...\
    # agent0_current_joint_pos = env.agent.agents[0].robot.get_qpos()[0, :-1].cpu().numpy()
    # env.step({"panda_wristcam-0": agent0_current_joint_pos, "panda_wristcam-1": agent1_target_joint_pos})
    planner.move_to_pose_with_screw(agent1_target_joint_pos, move_id=1)
    grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 0)
    grasp_poseA[2] += 0.04
    print(f"grasp_poseA: {grasp_poseA}")
    
    planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
    grasp_poseA[2] -= 0.04
    planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
    planner.close_gripper(0)
    target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    planner.move_to_pose_with_screw(target_poseA, move_id=0)
    planner.open_gripper(0)

    target_poseA[2] += 0.10
    planner.move_to_pose_with_screw(target_poseA, move_id=0)


    grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 0)
    grasp_poseB[2] += 0.04
    planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
    grasp_poseB[2] -= 0.04
    planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
    planner.close_gripper(0)
    grasp_poseB[2] += 0.10
    target_poseA = planner.get_grasp_pose_for_stack(grasp_poseB, env.cubeA)
    planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
    planner.move_to_pose_with_screw(target_poseA, move_id=0)
    planner.open_gripper(0)
    res = planner.open_gripper([0, 1])
    planner.close()
    return res
