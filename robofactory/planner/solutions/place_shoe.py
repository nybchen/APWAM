import numpy as np
import sapien

from robofactory.tasks import PlaceShoeEnv
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PlaceShoeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped
    grasp_pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe_left, actor_data=env.annotation_data['shoe_left'], pre_dis=0, id=0)

    res = planner.close_gripper([0, 1, 2])
    planner.close()
    return res

