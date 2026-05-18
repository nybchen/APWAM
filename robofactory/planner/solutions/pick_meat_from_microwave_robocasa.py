"""
Motion planning solution for pick_meat_from_microwave on RoboCasa.
XYZ mapped from Table + 180° rotation around pivot [2.2, -3.35, 0.92].
Quaternions rotated 180° around z.
"""
import numpy as np

from robofactory.tasks import PickMeatFromMicrowaveRobocasaEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

# Table agent_0 = [-0.85, -0.1, 0.], RoboCasa pivot = [2.2, -3.35, 0.92]
TABLE_TO_ROBOCASA_OFFSET = np.array([3.05, -3.25, 0.92])
PIVOT = np.array([2.2, -3.35, 0.92])  # 180° rotation center
X_SHIFT = 0.8  # Everything shifted +0.8 in x

# Quaternion for 180° around z (w,x,y,z)
Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions q1 * q2, format [w,x,y,z]."""
    w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return np.array([w, x, y, z])


def _map_pose(table_pose: np.ndarray) -> np.ndarray:
    """Map Table pose to RoboCasa: offset -> rotate 180° around pivot -> +0.8 x shift -> rotate quat."""
    xyz = table_pose[:3] + TABLE_TO_ROBOCASA_OFFSET
    xyz = np.array([2 * PIVOT[0] - xyz[0], 2 * PIVOT[1] - xyz[1], xyz[2]])
    xyz[0] += X_SHIFT
    quat = _quat_mul(Q_180Z, table_pose[3:7])
    return np.concatenate([xyz, quat])


def solve(env: PickMeatFromMicrowaveRobocasaEnv, seed=None, debug=False, vis=False):
    """
    Motion planning solution for picking meat from microwave (RoboCasa).
    Same sequence as Table; only xyz positions are mapped.
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

    # Robot 1 (door opener) - move to initial pose
    robot1_pose_table = np.array([-0.745552, -0.462443, 0.546143, -0.168495, 0.838735, 0.33362, 0.396019])
    robot1_pose = _map_pose(robot1_pose_table)
    planner.move_to_pose_with_screw(robot1_pose, move_id=1)

    # Robot 0 (perception) - top-down observation
    robot0_pose_table = np.array([-0.711235, 0.21192, 0.350585, 0.0214251, 0.980529, -0.0695068, 0.182406])
    robot0_pose = _map_pose(robot0_pose_table)
    planner.move_to_pose_with_screw(robot0_pose, move_id=0)

    # Phase 2: Robot 2 opens the microwave door (approach directions flipped for 180°)
    door_handle_grasp_pose = planner.get_grasp_pose_w_labeled_direction(
        actor=env.microwave,
        actor_data=env.annotation_data['microwave'],
        pre_dis=0,
        id=0
    )
    door_handle_grasp_pose[0] += 0.05
    door_handle_grasp_pose[2] += 0.02
    planner.move_to_pose_with_screw(door_handle_grasp_pose, move_id=2)

    door_handle_grasp_pose[0] -= 0.07
    planner.move_to_pose_with_screw(door_handle_grasp_pose, move_id=2)

    planner.close_gripper(close_id=[2])

    # Move to open the microwave door
    pose_table = np.array([-0.36139, -0.05641, 0.219849, 0.263025, 0.656391, -0.263059, 0.65633])
    pose = _map_pose(pose_table)
    planner.move_to_pose_with_screw(pose, move_id=2)

    pose_table = np.array([-0.462075, 0.162773, 0.219797, 0.514068, 0.485509, -0.514162, 0.48544])
    pose = _map_pose(pose_table)
    planner.move_to_pose_with_screw(pose, move_id=2)
    planner.open_gripper(open_id=[2])

    # Robot 2 moves to perception pose
    perception_pose_table = np.array([-0.52402, 0.409508, 0.544259, 0.243583, 0.73186, -0.528351, 0.354816])
    perception_pose = _map_pose(perception_pose_table)
    planner.move_to_pose_with_screw(perception_pose, move_id=2)

    # Robot 1 picks the meat (approach directions flipped for 180°)
    grasp_pose = planner.get_grasp_pose_w_labeled_direction(
        actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=2
    )
    grasp_pose[0] += 0.1
    grasp_pose[2] += 0.1
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    grasp_pose[2] -= 0.1
    grasp_pose[0] -= 0.11
    planner.move_to_pose_with_screw(grasp_pose, move_id=1)
    planner.close_gripper(close_id=[1])
    target_pose = planner.get_grasp_pose_for_stack(grasp_pose, env.goal_region)
    target_pose[2] += 0.1
    planner.move_to_pose_with_screw(target_pose, move_id=1)
    res = planner.open_gripper(open_id=[1])

    return res
