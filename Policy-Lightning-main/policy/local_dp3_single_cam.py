from typing import Dict, List
from omegaconf import DictConfig
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from policy.base_policy import BasePolicy
from model.diffusion.conditional_unet1d_3d import ConditionalUnet1D
from model.diffusion.mask_generator import LowdimMaskGenerator
from common.pytorch_util import dict_apply
from model.vision.pointnet_extractor import DP3Encoder
from common.pytorch_util import dict_apply


class DP3SingleCam(BasePolicy):
    """DP3 policy where each agent uses one assigned pointcloud source (observation agent pattern)."""

    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            optimazer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
            agent_num: int,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            share_obs_encoder: bool = False,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # Pointcloud assignment: list of pointcloud source IDs for each agent
            # e.g., [0, 1] means agent 0 uses pointcloud_0, agent 1 uses pointcloud_1
            # e.g., [1, 1] means both agents use pointcloud_1 (observation agent setup)
            pointcloud_assignment: List[int] = None,
            # parameters passed to step
            **kwargs):
        super().__init__(optimazer_cfg, scheduler_cfg)

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        # DP3Encoder expects agent_pos; alias from state if present
        if 'state' in obs_dict and 'agent_pos' not in obs_dict:
            obs_dict['agent_pos'] = obs_dict['state']

        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                img_crop_shape=crop_shape,
                                out_channel=encoder_output_dim,
                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                use_pc_color=use_pc_color,
                                pointnet_type=pointnet_type,
                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DP3SingleCam] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DP3SingleCam] pointnet_type: {self.pointnet_type}", "yellow")

        self.obs_encoders = nn.ModuleList(
            [deepcopy(obs_encoder) for _ in range(agent_num)]
        ) if not share_obs_encoder else obs_encoder
        self.model_list = nn.ModuleList(
            [ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        ) for _ in range(agent_num)]
        )
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.agent_num = agent_num
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Store pointcloud assignment
        if pointcloud_assignment is None:
            pointcloud_assignment = list(range(agent_num))
        assert len(pointcloud_assignment) == agent_num, \
            f"pointcloud_assignment length ({len(pointcloud_assignment)}) must match agent_num ({agent_num})"
        self.pointcloud_assignment = pointcloud_assignment
        cprint(f"[DP3SingleCam] pointcloud_assignment: {self.pointcloud_assignment}", "yellow")
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask, agent_id,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs):
        model = self.model_list[agent_id]
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
                
        trajectory[condition_mask] = condition_data[condition_mask]   
        return trajectory
    
    def get_local_agent_from_batch(self, batch, agent_id):
        """
        Get observations for an agent.
        - Use only the assigned pointcloud source for this agent (from pointcloud_assignment)
        - Use only this agent's own state
        """
        agent_obs = {}
        pc_source_id = self.pointcloud_assignment[agent_id]

        # Pointcloud from assigned source
        pc_key = f'pointcloud_{pc_source_id}'
        if pc_key in batch:
            agent_obs['pointcloud'] = batch[pc_key]

        # State from this agent's own
        state_key = f'state_{agent_id}'
        if state_key in batch:
            agent_obs['state'] = batch[state_key]

        # agent_pos mapping (DP3Encoder uses agent_pos as state_key)
        agent_pos_key = f'agent_pos_{agent_id}'
        if agent_pos_key in batch:
            agent_obs['agent_pos'] = batch[agent_pos_key]
        elif state_key in batch and 'agent_pos' not in agent_obs:
            agent_obs['agent_pos'] = batch[state_key]

        # Imagination robot from assigned source if present
        imagin_key = f'imagin_robot_{pc_source_id}'
        if imagin_key in batch:
            agent_obs['imagin_robot'] = batch[imagin_key]

        return agent_obs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """obs_dict: must include 'obs' key; result: must include 'action' key."""
        result = {}
        nobs = self.normalizer.normalize(obs_dict)
        
        for agent_id in range(self.agent_num):
            agent_nobs = self.get_local_agent_from_batch(nobs, agent_id)
            obs_encoder = self.obs_encoders[agent_id] if isinstance(self.obs_encoders, nn.ModuleList) else self.obs_encoders
        
            value = next(iter(agent_nobs.values()))
            B, To = value.shape[:2]
            T = self.horizon
            Da = self.action_dim
            Do = self.obs_feature_dim
            To = self.n_obs_steps

            device = self.device
            dtype = self.dtype

            local_cond = None
            global_cond = None
            if self.obs_as_global_cond:
                this_nobs = dict_apply(agent_nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                if "cross_attention" in self.condition_type:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
                else:
                    global_cond = nobs_features.reshape(B, -1)
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                this_nobs = dict_apply(agent_nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:,:To,Da:] = nobs_features
                cond_mask[:,:To,Da:] = True

            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                agent_id,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
            
            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer[f'action_{agent_id}'].unnormalize(naction_pred)

            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            
            result[f'action_{agent_id}'] = action
            result[f'action_pred_{agent_id}'] = action_pred
        
        return result

    def compute_loss(self, batch):
        total_loss = 0.0
        nobs = self.normalizer.normalize(batch['obs'])

        for agent_id in range(self.agent_num):
            model = self.model_list[agent_id]
            obs_encoder = self.obs_encoders[agent_id] if isinstance(self.obs_encoders, nn.ModuleList) else self.obs_encoders

            agent_nobs = self.get_local_agent_from_batch(nobs, agent_id)
            agent_action = batch[f'action_{agent_id}']
            nactions = self.normalizer[f'action_{agent_id}'].normalize(agent_action)
            batch_size = nactions.shape[0]
            horizon = nactions.shape[1]

            local_cond = None
            global_cond = None
            trajectory = nactions
            cond_data = trajectory
        
            if self.obs_as_global_cond:
                this_nobs = dict_apply(agent_nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)

                if "cross_attention" in self.condition_type:
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
                else:
                    global_cond = nobs_features.reshape(batch_size, -1)
            else:
                this_nobs = dict_apply(agent_nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
                trajectory = cond_data.detach()

            condition_mask = self.mask_generator(trajectory)
            noise = torch.randn(trajectory.shape, device=trajectory.device)

            bsz = trajectory.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()

            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timesteps)
            
            loss_mask = ~condition_mask
            noisy_trajectory[condition_mask] = cond_data[condition_mask]

            pred = model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                        local_cond=local_cond, 
                        global_cond=global_cond)

            pred_type = self.noise_scheduler.config.prediction_type 
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = trajectory
            elif pred_type == 'v_prediction':
                self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
                self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
                alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
                alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
                v_t = alpha_t * noise - sigma_t * trajectory
                target = v_t
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")

            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()

            total_loss += loss
        
        return total_loss / self.agent_num
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        obs_dict = batch['obs']
        total_mse = 0.0
        result = self.predict_action(obs_dict)
        for key, value in result.items():
            if key.startswith('action_pred'):
                pred_action = result[key]
                gt_action = batch[key.replace('_pred', '')]
                mse = F.mse_loss(pred_action, gt_action)
                total_mse += mse
        total_mse = total_mse / self.agent_num
        self.log('val/pred_action_mse', total_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss
