from typing import Any, Dict, Tuple

import copy
import os.path as osp

import numpy as np
import sapien
import torch
import yaml
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir


@register_env("TwoRobotsHandoverActive-rf", max_episode_steps=500)
class TwoRobotsHandoverActiveEnv(BaseEnv):
    """Two-arm cube handover task with mirrored directions and active perception."""

    default_config_name = "two_robots_handover_active.yaml"
    forced_direction = None
    goal_radius = 0.09
    cube_half_size_value = 0.02
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        if "config" in kwargs:
            with open(kwargs["config"], "r", encoding="utf-8") as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            del kwargs["config"]
        else:
            if "scene" in kwargs:
                scene = kwargs["scene"]
                del kwargs["scene"]
            else:
                scene = "table"
            with open(
                osp.join(CONFIG_DIR, scene, self.default_config_name),
                "r",
                encoding="utf-8",
            ) as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = nested_yaml_map(replace_dir, cfg)
        self.handover_direction = "left_to_right"
        self.source_agent_id = 0
        self.target_agent_id = 1
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
        sensor_cfg = cfg.get("cameras", {}).get("sensor", [])
        all_camera_configs = []
        for sensor in sensor_cfg:
            pose = sensor["pose"]
            if pose["type"] == "pose":
                sensor["pose"] = sapien.Pose(*pose["params"])
            elif pose["type"] == "look_at":
                sensor["pose"] = sapien_utils.look_at(*pose["params"])
            all_camera_configs.append(CameraConfig(**sensor))

        for idx, agent in enumerate(self.agent.agents):
            hand_camera_config = agent._sensor_configs[0]
            hand_camera_config.uid = f"hand_camera_{idx}"
            hand_camera_config.width = 320
            hand_camera_config.height = 240
            hand_camera_config.fov = 1.5707963268
            hand_camera_config.near = 0.01
            hand_camera_config.far = 10
            all_camera_configs.append(hand_camera_config)
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        cfg = copy.deepcopy(self.cfg)
        render_cfg = cfg.get("cameras", {}).get("human_render", [])
        all_camera_configs = []
        for render in render_cfg:
            pose = render["pose"]
            if pose["type"] == "pose":
                render["pose"] = sapien.Pose(*pose["params"])
            elif pose["type"] == "look_at":
                render["pose"] = sapien_utils.look_at(*pose["params"])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        init_poses = []
        for agent_cfg in cfg["agents"]:
            init_poses.append(sapien.Pose(p=agent_cfg["pos"]["ppos"]["p"]))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        for primitive_cfg in cfg["scene"].get("primitives", []):
            if primitive_cfg["name"] == "cube":
                self.cube_half_size_value = float(
                    primitive_cfg["params"].get("half_size", self.cube_half_size_value)
                )
                break
        self.cube_half_size = common.to_tensor(
            [self.cube_half_size_value] * 3, device=self.device
        )
        scene_name = cfg["scene"]["name"]
        scene_builder = getattr(scene_rf, f"{scene_name}SceneBuilder")
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

        direction = self.forced_direction or self.cfg.get("task", {}).get("direction", "random")
        if direction == "random":
            direction = "left_to_right" if np.random.rand() < 0.5 else "right_to_left"
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError(
                f"Unsupported handover direction {direction!r}; expected "
                "'left_to_right', 'right_to_left', or 'random'."
            )

        self.handover_direction = direction
        if direction == "left_to_right":
            self.source_agent_id = 0
            self.target_agent_id = 1
            source_goal = self.left_goal_region
        else:
            self.source_agent_id = 1
            self.target_agent_id = 0
            source_goal = self.right_goal_region

        cube_pose = self.cube.pose
        cube_pose.p[:, :2] = source_goal.pose.p[:, :2]
        cube_xy_noise = self.cfg.get("task", {}).get("cube_xy_noise", [0.0, 0.0])
        if isinstance(cube_xy_noise, (int, float)):
            cube_xy_noise = [float(cube_xy_noise), float(cube_xy_noise)]
        cube_xy_noise = torch.as_tensor(
            cube_xy_noise, device=self.device, dtype=cube_pose.p.dtype
        )
        if torch.any(cube_xy_noise > 0):
            xy_offset = (
                torch.rand(cube_pose.p[:, :2].shape, device=self.device) * 2.0 - 1.0
            ) * cube_xy_noise
            cube_pose.p[:, :2] += xy_offset
        cube_pose.p[:, 2] = self.cube_half_size_value
        self.cube.set_pose(cube_pose)
        cube_mass = self.cfg.get("task", {}).get("cube_mass")
        if cube_mass is not None:
            self.cube.set_mass(float(cube_mass))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def source_goal_region(self):
        return (
            self.left_goal_region
            if self.handover_direction == "left_to_right"
            else self.right_goal_region
        )

    @property
    def target_goal_region(self):
        return (
            self.right_goal_region
            if self.handover_direction == "left_to_right"
            else self.left_goal_region
        )

    def evaluate(self):
        cube_pos = self.cube.pose.p
        target_pos = self.target_goal_region.pose.p
        cube_to_target_dist = torch.linalg.norm(
            cube_pos[:, :2] - target_pos[:, :2], axis=1
        )
        cube_in_target = cube_to_target_dist < self.goal_radius
        is_left_grasped = self.left_agent.is_grasping(self.cube)
        is_right_grasped = self.right_agent.is_grasping(self.cube)
        success = cube_in_target * (~is_left_grasped) * (~is_right_grasped)
        source_agent_id = torch.full(
            cube_to_target_dist.shape,
            self.source_agent_id,
            device=self.device,
            dtype=torch.int64,
        )
        target_agent_id = torch.full(
            cube_to_target_dist.shape,
            self.target_agent_id,
            device=self.device,
            dtype=torch.int64,
        )
        return {
            "source_agent_id": source_agent_id,
            "target_agent_id": target_agent_id,
            "cube_to_target_dist": cube_to_target_dist,
            "cube_in_target": cube_in_target,
            "is_left_grasped": is_left_grasped,
            "is_right_grasped": is_right_grasped,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                left_goal_region_pos=self.left_goal_region.pose.p,
                right_goal_region_pos=self.right_goal_region.pose.p,
                source_goal_region_pos=self.source_goal_region.pose.p,
                target_goal_region_pos=self.target_goal_region.pose.p,
                cube_pose=self.cube.pose.raw_pose,
                left_arm_tcp_to_cube_pos=self.cube.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cube_pos=self.cube.pose.p - self.right_agent.tcp.pose.p,
                cube_to_target_goal_pos=self.target_goal_region.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
