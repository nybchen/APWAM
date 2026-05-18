from typing import Any, Dict, Tuple

import os.path as osp
import numpy as np
import sapien
import torch
import yaml
import copy

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam


@register_env("PickCubeFromCabinet-rf", max_episode_steps=500)
class PickCubeFromCabinetEnv(BaseEnv):
    """
    Three robots task for picking cube from cabinet with active perception.
    
    Robot 0: Active perception (look at the whole scene from top)
    Robot 1: Open the cabinet drawer
    Robot 2: Pick cube and place to goal region
    """
    
    goal_radius = 0.12
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam, PandaWristCam]]

    def __init__(
        self, *args, robot_uids=("panda_wristcam", "panda_wristcam", "panda_wristcam"), robot_init_qpos_noise=0.02, **kwargs
    ):
        if 'config' in kwargs:
            with open(kwargs['config'], 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            del kwargs['config']
        else:
            if 'scene' in kwargs:
                scene = kwargs['scene']
                del kwargs['scene']
            else:
                scene = 'table'
            with open(osp.join(CONFIG_DIR, scene, 'pick_cube_from_cabinet.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = nested_yaml_map(replace_dir, cfg)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs = []
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if pose['type'] == 'pose':
                sensor['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                sensor['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**sensor))
        # add hand camera for each agent
        temp = 0
        for agent in self.agent.agents:
            hand_camera_config = agent._sensor_configs[0]
            hand_camera_config.uid = f'hand_camera_{temp}'
            hand_camera_config.width = 320
            hand_camera_config.height = 240
            hand_camera_config.fov = 1.5707963268
            hand_camera_config.near = 0.01
            hand_camera_config.far = 10
            all_camera_configs.append(hand_camera_config)
            temp += 1
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs = []
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        init_poses = []
        for agent_cfg in cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        scene_name = cfg['scene']['name']
        scene_builder = getattr(scene_rf, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

    @property
    def perception_agent(self) -> PandaWristCam:
        """Robot 0: Active perception (look at the whole scene from top)"""
        return self.agent.agents[0]

    @property
    def drawer_opener_agent(self) -> PandaWristCam:
        """Robot 1: Open the cabinet drawer"""
        return self.agent.agents[1]

    @property
    def picker_agent(self) -> PandaWristCam:
        """Robot 2: Pick cube and place to goal"""
        return self.agent.agents[2]

    def evaluate(self):
        cube_pose = self.cube.pose.p
        goal_pose = self.goal_region.pose.p
        
        # Check if cube is placed in goal region
        cube_to_goal_dist = torch.linalg.norm(
            cube_pose[:, :2] - goal_pose[:, :2], axis=1
        )
        cube_in_goal = cube_to_goal_dist < self.goal_radius
        
        # Check if cube is on the table (not in drawer)
        cube_on_table = cube_pose[:, 2] < 0.1
        
        success = cube_in_goal & cube_on_table
        success = success.bool()
        return {
            "cube_to_goal_dist": cube_to_goal_dist,
            "cube_in_goal": cube_in_goal,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            perception_tcp=self.perception_agent.tcp.pose.raw_pose,
            drawer_opener_tcp=self.drawer_opener_agent.tcp.pose.raw_pose,
            picker_tcp=self.picker_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                goal_region_pos=self.goal_region.pose.p,
                cabinet_pose=self.cabinet.pose.raw_pose,
                cube_pose=self.cube.pose.raw_pose,
                perception_tcp_to_cabinet=self.cabinet.pose.p - self.perception_agent.tcp.pose.p,
                drawer_opener_tcp_to_cabinet=self.cabinet.pose.p - self.drawer_opener_agent.tcp.pose.p,
                picker_tcp_to_cube=self.cube.pose.p - self.picker_agent.tcp.pose.p,
                cube_to_goal=self.goal_region.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

