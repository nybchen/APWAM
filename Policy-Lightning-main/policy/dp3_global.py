from typing import Dict

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


class DP3Global(BasePolicy):
    """
    Multi-agent DP3 variant that:
    - Each agent sees ALL pointclouds (concatenated) + own state,
    - Uses a single 1D UNet to model the joint action trajectory for all agents.

    Expected data layout:
    - batch['obs'] contains keys like 'pointcloud_0', 'pointcloud_1', 'state_0', 'state_1', ...
    - batch contains per-agent actions: 'action_0', 'action_1', ...
    The network predicts all agents' actions jointly but returns them split back out as
    'action_{i}' and 'action_pred_{i}' for compatibility with existing code.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        optimazer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        agent_num: int,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        share_obs_encoder: bool = False,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        encoder_output_dim: int = 256,
        crop_shape=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        pointcloud_encoder_cfg=None,
        # parameters passed to step / scheduler.step
        **kwargs,
    ):
        super().__init__(optimazer_cfg, scheduler_cfg)

        self.condition_type = condition_type
        self.agent_num = agent_num

        # ===== parse action shape =====
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            per_agent_action_dim = action_shape[0]
        elif len(action_shape) == 2:  # e.g. multiple hands
            per_agent_action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        self.action_dim_per_agent = per_agent_action_dim
        # Total action dim is concatenation of all agents
        action_dim = per_agent_action_dim * agent_num
        self.action_dim = action_dim

        # ===== parse obs shape and build encoders =====
        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])
        # DP3Encoder expects agent_pos; alias from state if present
        if "state" in obs_dict and "agent_pos" not in obs_dict:
            obs_dict["agent_pos"] = obs_dict["state"]

        base_obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        obs_feature_dim_single = base_obs_encoder.output_shape()
        self.obs_feature_dim_single = obs_feature_dim_single
        # Concatenate features from all agents
        obs_feature_dim = obs_feature_dim_single * agent_num
        self.obs_feature_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DP3Global] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DP3Global] pointnet_type: {self.pointnet_type}", "yellow")

        # One encoder per agent (or shared)
        self.obs_encoders = (
            nn.ModuleList([deepcopy(base_obs_encoder) for _ in range(agent_num)])
            if not share_obs_encoder
            else base_obs_encoder
        )

        # ===== create diffusion model (single UNet for all agents) =====
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            # obs only enters as global condition
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                # sequence of features over time
                global_cond_dim = obs_feature_dim
            else:
                # flattened sequence of features
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
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
        )

        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= helpers ============
    def get_local_agent_from_batch(self, batch: Dict[str, torch.Tensor], agent_id: int):
        """
        Get observations for an agent.
        - Concatenate all pointclouds from all cameras (along point dimension)
        - Use only this agent's own state
        """
        agent_obs = {}

        all_pointclouds = []
        for other_agent_id in range(self.agent_num):
            pc_key = f"pointcloud_{other_agent_id}"
            if pc_key in batch:
                all_pointclouds.append(batch[pc_key])

        if all_pointclouds:
            stacked_pc = torch.cat(all_pointclouds, dim=2)
            agent_obs["pointcloud"] = stacked_pc

        state_key = f"state_{agent_id}"
        agent_pos_key = f"agent_pos_{agent_id}"
        if state_key in batch:
            agent_obs["state"] = batch[state_key]
        if agent_pos_key in batch:
            agent_obs["agent_pos"] = batch[agent_pos_key]
        if "state" in agent_obs and "agent_pos" not in agent_obs:
            agent_obs["agent_pos"] = agent_obs["state"]

        return agent_obs

    def _encode_global_obs_features(
        self,
        nobs: Dict[str, torch.Tensor],
        B: int,
        To: int,
        use_cross_attention: bool,
    ):
        """
        Encode all agents' observations into a single global feature tensor.

        Returns:
            - if use_cross_attention: Tensor of shape [B, To, obs_feature_dim]
            - else: Tensor of shape [B, To, obs_feature_dim]
        (The caller decides whether to flatten over time or not.)
        """
        per_agent_features = []
        for agent_id in range(self.agent_num):
            agent_nobs = self.get_local_agent_from_batch(nobs, agent_id)
            if len(agent_nobs) == 0:
                raise RuntimeError(
                    f"No observation keys found for agent {agent_id} in DP3Global."
                )
            obs_encoder = (
                self.obs_encoders[agent_id]
                if isinstance(self.obs_encoders, nn.ModuleList)
                else self.obs_encoders
            )
            this_nobs = dict_apply(
                agent_nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = obs_encoder(this_nobs)  # (B*To, obs_feature_dim_single)
            nobs_features = nobs_features.reshape(B, To, -1)  # (B, To, obs_feature_dim_single)
            per_agent_features.append(nobs_features)

        # Concatenate all agents along feature dimension
        global_features = torch.cat(per_agent_features, dim=-1)  # (B, To, obs_feature_dim)
        return global_features

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous sample: x_t -> x_{t-1}
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key when called from training loop,
                  but DeployPolicy passes a flat dict of obs tensors.

        Returns:
            Dict with keys 'action_i' and 'action_pred_i' for each agent i.
        """
        result: Dict[str, torch.Tensor] = {}

        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)

        # Find a sample obs tensor to read B, To from
        sample_agent_nobs = None
        for agent_id in range(self.agent_num):
            agent_nobs = self.get_local_agent_from_batch(nobs, agent_id)
            if len(agent_nobs) > 0:
                sample_agent_nobs = agent_nobs
                break
        if sample_agent_nobs is None:
            raise RuntimeError("DP3Global.predict_action: no agent observations found.")

        value = next(iter(sample_agent_nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da_single = self.action_dim_per_agent
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            # condition through global feature for all agents jointly
            global_features = self._encode_global_obs_features(
                nobs, B=B, To=To, use_cross_attention="cross_attention" in self.condition_type
            )
            if "cross_attention" in self.condition_type:
                # treat as a sequence [B, To, feat_dim]
                global_cond = global_features
            else:
                # flatten over time
                global_cond = global_features.reshape(B, -1)

            # empty data for action (will be sampled)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through inpainting: append obs features to trajectory
            global_features = self._encode_global_obs_features(
                nobs, B=B, To=To, use_cross_attention=False
            )  # (B, To, obs_feature_dim)
            Do = global_features.shape[-1]
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = global_features
            cond_mask[:, :To, Da:] = True

        # run sampling once for all agents
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction and split back to per-agent actions
        naction_pred_all = nsample[..., :Da]  # (B, T, Da_total)

        for agent_id in range(self.agent_num):
            start_idx = agent_id * Da_single
            end_idx = (agent_id + 1) * Da_single
            naction_pred = naction_pred_all[..., start_idx:end_idx]
            action_pred = self.normalizer[f"action_{agent_id}"].unnormalize(naction_pred)

            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

            result[f"action_{agent_id}"] = action
            result[f"action_pred_{agent_id}"] = action_pred

        return result

    # ========= training ============
    def compute_loss(self, batch):
        # normalize obs
        nobs = self.normalizer.normalize(batch["obs"])

        # build joint normalized action trajectory for all agents
        nactions_list = []
        for agent_id in range(self.agent_num):
            agent_action = batch[f"action_{agent_id}"]
            nactions = self.normalizer[f"action_{agent_id}"].normalize(agent_action)
            nactions_list.append(nactions)

        nactions_all = torch.cat(nactions_list, dim=-1)  # (B, T, Da_total)
        batch_size = nactions_all.shape[0]
        horizon = nactions_all.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions_all
        cond_data = trajectory

        if self.obs_as_global_cond:
            # encode observations for all agents and condition globally
            global_features = self._encode_global_obs_features(
                nobs,
                B=batch_size,
                To=self.n_obs_steps,
                use_cross_attention="cross_attention" in self.condition_type,
            )

            if "cross_attention" in self.condition_type:
                global_cond = global_features  # (B, To, feat_dim)
            else:
                global_cond = global_features.reshape(batch_size, -1)
        else:
            # concatenate obs features to trajectory for inpainting-style conditioning
            global_features = self._encode_global_obs_features(
                nobs,
                B=batch_size,
                To=horizon,
                use_cross_attention=False,
            )  # (B, T, feat_dim)
            cond_data = torch.cat([nactions_all, global_features], dim=-1)
            trajectory = cond_data.detach()

        # generate inpainting mask
        condition_mask = self.mask_generator(trajectory)

        # Sample noise that we'll add to the trajectories
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each sample in the batch
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        # Forward diffusion process
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise / sample residual
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            # See diffusers' v_prediction docs
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        obs_dict = batch["obs"]
        total_mse = 0.0
        result = self.predict_action(obs_dict)
        for key, value in result.items():
            if key.startswith("action_pred"):
                pred_action = result[key]
                gt_action = batch[key.replace("_pred", "")]
                mse = F.mse_loss(pred_action, gt_action)
                total_mse += mse
        total_mse = total_mse / self.agent_num
        self.log(
            "val/pred_action_mse",
            total_mse,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

