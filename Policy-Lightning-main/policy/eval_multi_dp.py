import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import fpsample
import torch  
import os

import hydra
from pathlib import Path
from collections import deque, defaultdict
from tasks import *
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner
from planner.motionplanner import PandaArmMotionPlanningSolver
from diffusion_policy.model.noposplat.encoder import get_encoder
from lightning.pytorch import LightningModule

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from utils.wrappers.record import RecordEpisodeMA
from utils.wrappers.suction_action import SuctionActionWrapper

import tyro
from dataclasses import dataclass
from omegaconf import OmegaConf 
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "configs/table/place_badminton.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = './testvideo/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 1000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    data_num: int = 100
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint"""

    record_dir: Optional[str] = './eval_video/DP/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 100
    """Maximum number of steps to run the simulation"""

    ckpt: str = '/home/martel/CodeSpace/RoboFactory/RoboFactory/checkpoints/pb25_300.ckpt'

    exp_name: str = ''

def get_policy(checkpoint, output_dir, device):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = OmegaConf.create(cfg)
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload['state_dict'])
    device = torch.device(device)
    policy = model.to(device)
    policy.eval()
    return policy

class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, ckpt_path, id: int = 0):
        self.policy = get_policy(ckpt_path, None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def init_runner(self):
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)

    def get_last_obs(self):
        return self.runner.obs[-1]
    
    def reset_policy(self):
        self.policy.reset()

def get_model_input(observation, agent_pos_list, agent_num):
    obs = {}
    camera_name = 'head_camera_global'
    head_cam = np.moveaxis(observation['sensor_data'][camera_name]['rgb'].squeeze(0).cpu().numpy(), -1, 0) / 255   
    obs.update({f'head_cam': head_cam})

    agent_pos = []
    for agent_id in range(agent_num):
        agent_pos.append(agent_pos_list[agent_id])
    obs.update({f'agent_pos': np.concatenate(agent_pos, axis=-1)})
    return obs

def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = 0
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'
            print(env_id)
    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/dp2_{args.exp_name}_{env_id}_{args.data_num}_{args.checkpoint_num}_{timestamp}.txt"

    dp = DP(env_id, args.checkpoint_num, args.data_num, args.ckpt)
    total_success = 0
    total_num = 0
    now_success = 0
    now_test = f"test/dp2/{env_id}"
    for now_seed in range(args.seed[0], args.seed[0] + 100):
        seed_folder = os.path.join(now_test, str(now_seed))
        dp.init_runner()
        dp.reset_policy()
        env: BaseEnv = gym.make(env_id, **env_kwargs)
        print("Current eval seed: ", now_seed)
        total_num += 1
        now_success = 0
        np.random.seed(now_seed)

        env = SuctionActionWrapper(
            env,
            on_threshold=0.0,
            # suction_agents=['panda-0'],
            stick_agents=['panda_stick-1'],
            # circle_agents=['__single__'],
            gripper_hold_value=-1.0,
            probe_internal_steps=3,             
            debug=True
        )
        tool = getattr(env.unwrapped, "_shared_suction_tool", None)
        assert tool is not None, "SuctionTool not found; make sure SuctionActionWrapper is applied."
        tool.clear_filters()
        tool.allow_only_names(contains=["card"])          
        tool.disallow_names(contains=["table", "floor", "ground", "cube"])

        record_dir = args.record_dir + f'_dp2_{args.exp_name}_' + str(timestamp) + '/' + str(now_seed)
        if record_dir:
            record_dir = record_dir.format(env_id=env_id)
            env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000000)

        raw_obs, _ = env.reset(seed=now_seed)
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=verbose,
            base_pose=[agent.robot.pose for agent in env.agent.agents],
            visualize_target_grasp_pose=verbose,
            print_env_info=False,
            is_multi_agent=True,
            tool_modes=['gripper', 'stick']
        )
        agent_num = planner.agent_num
        if now_seed is not None and env.action_space is not None:
            env.action_space.seed(now_seed)
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()

        spaces_is_dict = isinstance(env.action_space, gym.spaces.Dict)
        space_keys = list(env.action_space.spaces.keys()) if spaces_is_dict else []
        obs_keys = list(raw_obs['agent'].keys())
        try:
            uid_list = [ag.uid for ag in env.agent.agents]
        except Exception:
            uid_list = obs_keys

        def _base(s):
            return s.split('-', 1)[0] if isinstance(s, str) else s

        agent_keys = []
        used = set()
        for idx in range(agent_num):
            uid = uid_list[idx] if idx < len(uid_list) else None
            k = uid if uid in obs_keys and uid not in used else None
            if k is None:
                b = _base(uid) if uid is not None else None
                k = next((x for x in obs_keys if _base(x) == b and x not in used), None)
            if k is None:
                k = obs_keys[idx if idx < len(obs_keys) else -1]
            agent_keys.append(k)
            used.add(k)

        if spaces_is_dict:
            arm_dofs_map = {}
            for i, k in enumerate(agent_keys):
                ks = k if k in env.action_space.spaces else next((x for x in space_keys if _base(x) == _base(k)), None)
                if ks is None:
                    ks = space_keys[i if i < len(space_keys) else 0]
                arm_dofs_map[k] = int(env.action_space.spaces[ks].shape[0] - 1)
        else:
            agent_keys = ["__single__"]
            arm_dofs_map = {"__single__": int(env.action_space.shape[0] - 1)}

        initial_qpos_list = []
        for id in range(agent_num):
            key = agent_keys[id]
            q = raw_obs['agent'][key]['qpos'].squeeze(0)
            q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
            qpos_arm = q_arr[:arm_dofs_map[key]]
            initial_qpos = np.append(qpos_arm, planner.gripper_state[id])
            initial_qpos_list.append(initial_qpos)

        obs = get_model_input(raw_obs, initial_qpos_list, agent_num)
        dp.update_obs(obs)
        cnt = 0

        while True:
            if verbose:
                print("Iteration: ", cnt)
            cnt += 1
            if cnt > args.max_steps:
                break
            if cnt % 15 == 0:
                print("iter: ", cnt)

            action = dp.get_action()
            action_dict = defaultdict(list)
            action_step_dict = defaultdict(list)

            for id in range(agent_num):
                key = agent_keys[id]
                action_list = []
                for t in range(len(action[f'action_{id}'])):
                    agent_action = action[f'action_{id}'][t]
                    action_list.append(agent_action)

                for i in range(8):
                    now_action = action_list[i]
                    raw_obs = env.get_obs()
                    if i == 0:
                        q = raw_obs['agent'][key]['qpos'].squeeze(0)
                        q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                        current_qpos = q_arr[:arm_dofs_map[key]]
                    else:
                        current_qpos = action_list[i - 1][:-1]
                    path = np.vstack((current_qpos, now_action[:-1]))
                    try:
                        times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.05, verbose=True)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        action_now = np.hstack([current_qpos, now_action[-1]])
                        action_dict[key].append(action_now)
                        action_step_dict[key].append(1)
                        continue
                    n_step = position.shape[0]
                    action_step_dict[key].append(n_step)
                    gripper_state = now_action[-1]
                    if n_step == 0:
                        action_now = np.hstack([current_qpos, gripper_state])
                        action_dict[key].append(action_now)
                    for j in range(n_step):
                        true_action = np.hstack([position[j], gripper_state])
                        action_dict[key].append(true_action)
            
            start_idx = [0 for _ in range(agent_num)]
            for i in range(8):
                max_step = 0
                for id in range(agent_num):
                    key = agent_keys[id]
                    max_step = max(max_step, action_step_dict[key][i])
                for j in range(max_step):
                    true_action = dict()
                    for id in range(agent_num):
                        key = agent_keys[id]
                        now_step = min(j, action_step_dict[key][i] - 1)
                        true_action[key] = action_dict[key][start_idx[id] + now_step]
                    observation, reward, terminated, truncated, info = env.step(true_action)

                if max_step == 0:
                    continue
                action_concat = []
                for id in range(agent_num):
                    key = agent_keys[id]
                    start_idx[id] += action_step_dict[key][i]
                    action_concat.append(true_action[key])

                if action_concat:
                    obs = get_model_input(observation, action_concat, agent_num)
                    dp.update_obs(obs)

            info = env.get_info()
            if args.render_mode is not None:
                env.render()
            if info['success'] == True:
                total_success += 1
                now_success = 1
                env.close()
                if record_dir:
                    print(f"Saving video to {record_dir}")
                print("success, step=", cnt)
                os.makedirs(seed_folder, exist_ok=True)
                break

        with open(log_file, "a") as f:
            f.write(f"\n[Summary] Success Rate: {total_success}% / {total_num}\n")
            f.write(f"Current Seeds: {now_seed}, success: {now_success}\n")
        if now_success == 0:
            print("failed")
            os.makedirs(seed_folder, exist_ok=True)
            env.close() 
        if record_dir:
            print(f"Saving video to {record_dir}")

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
