import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 
sys.path.insert(0, '/home/nybchen/MARS/MAAP/Policy-Lightning-main') 
sys.path.insert(0, '/home/nybchen/MARS/MAAP/robofactory') 

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


import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.visualization.misc import images_to_video
from utils.wrappers.record import RecordEpisodeMA

from omegaconf import OmegaConf 
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "/home/nybchen/MARS/MAAP/robofactory/configs/table/two_robots_stack_cube_active.yaml"
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
    """Render mode - set to rgb_array to avoid GUI"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = './testvideo/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    data_num: int = 150
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint"""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 250
    """Maximum number of steps to run the simulation"""
    
    ckpt_path: str = '/home/bingxing2/ailab/kangli/brunotemp/MAAP/Policy-Lightning-main/outputs/Local_DP2/2025.11.29.06.22.40_2a_two_robots_stack_cube_active_local/checkpoints/epoch=299-loss=0.0001.ckpt'

# def get_policy(checkpoint, output_dir, device):
#     # load checkpoint
#     payload = torch.load(open('./'+checkpoint, 'rb'), pickle_module=dill)
#     cfg = payload['cfg']
#     cls = hydra.utils.get_class(cfg._target_)
#     workspace = cls(cfg, output_dir=output_dir)
#     workspace: RobotWorkspace
#     workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
#     # get policy from workspace
#     policy = workspace.model
#     if cfg.training.use_ema:
#         policy = workspace.ema_model
    
#     device = torch.device(device)
#     policy.to(device)
#     policy.eval()

#     return policy

def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # checkpoint_dir = os.path.dirname(checkpoint)
    # config_path = f"{checkpoint_dir}/../.hydra/config.yaml"
    # cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.create(cfg)
    # if cfg.encoder.pretrained_weights:
    #     weight_path = cfg.encoder.pretrained_weights
    #     ckpt_weights = torch.load(weight_path, map_location="cpu", weights_only=False)
    #     ckpt_weights = ckpt_weights["state_dict"]
    #     missing_keys, unexpected_keys = gaussian_encoder.load_state_dict(ckpt_weights)
    #     print("successfully loaded encoder weights")
    # else:
    #     raise ValueError(f"Invalid checkpoint format: {weight_path}")

    # configure model
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload['state_dict'], strict=False)

    device = torch.device(device)
    policy = model.to(device)
    policy.eval()

    return policy

class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, ckpt_path, agent_num: int):
        self.policy = get_policy(ckpt_path, None, 'cuda:0')
        self.agent_num = agent_num
        # Create separate runners for each agent
        self.runners = [DPRunner(output_dir=None) for _ in range(agent_num)]

    def init_runners(self):
        self.runners = [DPRunner(output_dir=None) for _ in range(self.agent_num)]

    def update_obs(self, observation, agent_id):
        self.runners[agent_id].update_obs(observation)
    
    def get_action(self):
        # Collect observations from all agents
        # The policy expects head_cam_0, head_cam_1, state_0, state_1 directly (not wrapped in 'obs')
        # It will stack cameras side-by-side internally for each agent
        all_obs_dict = {}
        for agent_id in range(self.agent_num):
            obs = self.runners[agent_id].get_n_steps_obs()
            # Add agent_id suffix to keys to match what policy expects
            for key, value in obs.items():
                if key == 'head_cam':
                    all_obs_dict[f'head_cam_{agent_id}'] = value
                elif key == 'state':
                    all_obs_dict[f'state_{agent_id}'] = value
                else:
                    all_obs_dict[f'{key}_{agent_id}'] = value
        
        # device transfer
        device, dtype = self.policy.device, self.policy.dtype
        obs_dict_tensor = dict_apply(all_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        
        # run policy - pass all agents' observations at once
        # The policy's normalizer expects keys directly (head_cam_0, head_cam_1, state_0, state_1)
        with torch.no_grad():
            obs_dict_input = {}
            for key in obs_dict_tensor.keys():
                if key.startswith('head_cam') or key.startswith('state'):          
                    obs_dict_input[key] = obs_dict_tensor[key].unsqueeze(0)
            action_dict = self.policy.predict_action(obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        
        # Extract actions for all agents
        # Policy returns action_{agent_id} with shape [B, T, D] where B=1, T=n_action_steps, D=action_dim
        actions = {}
        for agent_id in range(self.agent_num):
            action_key = f'action_{agent_id}'
            if action_key in np_action_dict:
                actions[agent_id] = np_action_dict[action_key].squeeze(0)  # Remove batch dimension: [T, D]
            else:
                raise KeyError(f"Could not find action key '{action_key}' in policy output")
        
        return actions

    def get_last_obs(self, agent_id):
        return self.runners[agent_id].obs[-1]
    
def get_model_input(observation, agent_pos, agent_id):
    camera_name = 'hand_camera_' + str(agent_id)
    hand_cam = np.moveaxis(observation['sensor_data'][camera_name]['rgb'].squeeze(0).numpy(), -1, 0) / 255
    # The policy expects head_cam_0, head_cam_1, state_0, state_1
    # It will stack cameras side-by-side internally for each agent
    return dict(
        head_cam = hand_cam,  # Will be used as head_cam_{agent_id} by the model
        state = agent_pos,  # Will be used as state_{agent_id} by the model
    )

def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = False  # Disable verbose output
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = False  # Disable GUI
    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'
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
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    record_dir = args.record_dir + '/' + str(args.seed) + '_' + str(args.data_num) + '_' + str(args.checkpoint_num)
    if record_dir:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000)

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True
    )

    # Load multi dp policy - single instance for all agents
    agent_num = planner.agent_num
    dp_model = DP(env_id, args.checkpoint_num, args.data_num, args.ckpt_path, agent_num)

    raw_obs, _ = env.reset(seed=args.seed)
    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    
    # Initialize lists to store hand camera frames for recording
    hand_cam_frames = [[] for _ in range(agent_num)]  # One list per agent
    
    # Capture initial hand camera frames
    if record_dir and 'sensor_data' in raw_obs:
        for agent_id in range(agent_num):
            camera_name = f'hand_camera_{agent_id}'
            if camera_name in raw_obs['sensor_data']:
                cam_rgb = raw_obs['sensor_data'][camera_name]['rgb']
                if isinstance(cam_rgb, torch.Tensor):
                    cam_rgb = cam_rgb.cpu().numpy()
                if len(cam_rgb.shape) == 4:  # [1, H, W, 3]
                    cam_rgb = cam_rgb.squeeze(0)
                if cam_rgb.dtype != np.uint8:
                    cam_rgb = (cam_rgb * 255).astype(np.uint8) if cam_rgb.max() <= 1.0 else cam_rgb.astype(np.uint8)
                hand_cam_frames[agent_id].append(cam_rgb)
    
    for id in range(agent_num):
        initial_qpos = raw_obs['agent'][f'panda_wristcam-{id}']['qpos'].squeeze(0)[:-2].numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[id])
        obs = get_model_input(raw_obs, initial_qpos, id)
        dp_model.update_obs(obs, id)
    
    cnt = 0
    eval_count = 0
    print(f"Starting evaluation...")
    while True:
        cnt = cnt + 1
        if cnt > args.max_steps:
            break
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        # Get actions for all agents at once
        all_actions = dp_model.get_action()
        for id in range(agent_num):
            action_list = all_actions[id]
            for i in range(6):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'panda_wristcam-{id}']['qpos'].squeeze(0)[:-2].numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.05, verbose=False)
                except Exception as e:
                    action_now = np.hstack([current_qpos, now_action[-1]])
                    action_dict[f'panda_wristcam-{id}'].append(action_now)
                    action_step_dict[f'panda_wristcam-{id}'].append(1)
                    continue
                n_step = position.shape[0]
                action_step_dict[f'panda_wristcam-{id}'].append(n_step)
                gripper_state = now_action[-1]
                if n_step == 0:
                    action_now = np.hstack([current_qpos, gripper_state])
                    action_dict[f'panda_wristcam-{id}'].append(action_now)
                for j in range(n_step):
                    true_action = np.hstack([position[j], gripper_state])
                    action_dict[f'panda_wristcam-{id}'].append(true_action)
        
        start_idx = []
        for id in range(agent_num):
            start_idx.append(0)
        for i in range(6):
            max_step = 0
            for id in range(agent_num):
                max_step = max(max_step, action_step_dict[f'panda_wristcam-{id}'][i])
            for j in range(max_step):
                true_action = dict()
                for id in range(agent_num):
                    now_step = min(j, action_step_dict[f'panda_wristcam-{id}'][i] - 1)
                    true_action[f'panda_wristcam-{id}'] = action_dict[f'panda_wristcam-{id}'][start_idx[id] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                
                # Capture hand camera frames for recording
                if record_dir and 'sensor_data' in observation:
                    for agent_id in range(agent_num):
                        camera_name = f'hand_camera_{agent_id}'
                        if camera_name in observation['sensor_data']:
                            # Get RGB image: shape is [1, H, W, 3] or [H, W, 3]
                            cam_rgb = observation['sensor_data'][camera_name]['rgb']
                            if isinstance(cam_rgb, torch.Tensor):
                                cam_rgb = cam_rgb.cpu().numpy()
                            # Handle different shapes
                            if len(cam_rgb.shape) == 4:  # [1, H, W, 3]
                                cam_rgb = cam_rgb.squeeze(0)
                            # Ensure uint8 format [H, W, 3]
                            if cam_rgb.dtype != np.uint8:
                                cam_rgb = (cam_rgb * 255).astype(np.uint8) if cam_rgb.max() <= 1.0 else cam_rgb.astype(np.uint8)
                            hand_cam_frames[agent_id].append(cam_rgb)
                            
            for id in range(agent_num):
                start_idx[id] += action_step_dict[f'panda_wristcam-{id}'][i]
                if action_step_dict[f'panda_wristcam-{id}'][i] == 0:
                    continue
                obs = get_model_input(observation, true_action[f'panda_wristcam-{id}'], id)
                dp_model.update_obs(obs, id)
        
        # Save hand camera videos if we have frames
        if record_dir and len(hand_cam_frames[0]) > 0:
            record_path = Path(record_dir)
            record_path.mkdir(parents=True, exist_ok=True)
            for agent_id in range(agent_num):
                if len(hand_cam_frames[agent_id]) > 0:
                    video_name = f"hand_camera_{agent_id}"
                    images_to_video(
                        hand_cam_frames[agent_id],
                        str(record_path),
                        video_name=video_name,
                        fps=30,
                        verbose=False,
                    )
                    print(f"Saved hand camera video for agent {agent_id}: {record_path}/{video_name}.mp4")
        
        if info['success'] == True:
            eval_count += 1
            print(f"Evaluated: {eval_count} (Success)")
            env.close()
            return
    eval_count += 1
    print(f"Evaluated: {eval_count} (Failed)")
    
    # Save hand camera videos even on failure
    if record_dir and len(hand_cam_frames[0]) > 0:
        record_path = Path(record_dir)
        record_path.mkdir(parents=True, exist_ok=True)
        for agent_id in range(agent_num):
            if len(hand_cam_frames[agent_id]) > 0:
                video_name = f"hand_camera_{agent_id}"
                images_to_video(
                    hand_cam_frames[agent_id],
                    str(record_path),
                    video_name=video_name,
                    fps=30,
                    verbose=False,
                )
                print(f"Saved hand camera video for agent {agent_id}: {record_path}/{video_name}.mp4")
    
    env.close()

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
