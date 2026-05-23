from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModeNetConfig:
    """Configuration for language-conditioned online MAAP allocation."""

    max_agents: int
    q_dim: int
    action_dim: int
    image_encoder: str = "dinov2_vits14"
    image_feature_dim: Optional[int] = None
    freeze_image_encoder: bool = True
    dino_repo: str = "facebookresearch/dinov2"
    dino_source: str = "github"
    dino_pretrained: bool = True
    dino_image_size: int = 224
    normalize_images: bool = True
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_history: int = 16
    max_language_tokens: int = 64
    vocab_size: Optional[int] = None
    num_tasks: Optional[int] = None
    image_channels: int = 3
    language_dim: Optional[int] = None
    output_activation: str = "sigmoid"


class SmallImageEncoder(nn.Module):
    """Compact image encoder for global and wrist-camera history tokens."""

    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


class DinoV2ImageEncoder(nn.Module):
    """DINOv2 image encoder projected into ModeNet's token dimension.

    The backbone is loaded through torch.hub so we do not add a transformers or
    timm dependency. In offline environments, set ``image_encoder="small_cnn"``
    for tests, or make sure the torch hub repo and weights are already cached.
    """

    FEATURE_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14_reg": 1536,
    }

    def __init__(
        self,
        *,
        model_name: str,
        d_model: int,
        feature_dim: Optional[int],
        repo: str,
        source: str,
        pretrained: bool,
        freeze: bool,
        image_size: int,
        normalize_images: bool,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.freeze = freeze
        self.backbone = torch.hub.load(
            repo,
            model_name,
            pretrained=pretrained,
            source=source,
            trust_repo=True,
        )
        if freeze:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

        in_dim = feature_dim or self.FEATURE_DIMS.get(model_name)
        if in_dim is None:
            raise ValueError(
                "image_feature_dim must be set for non-standard DINOv2 backbones"
            )
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = self._preprocess(images)
        if self.freeze:
            with torch.no_grad():
                feat = self._forward_backbone(images)
        else:
            feat = self._forward_backbone(images)
        return self.proj(feat)

    def _preprocess(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        if images.shape[1] != 3:
            raise ValueError("DINOv2 encoder expects 3-channel RGB images")
        was_uint8 = images.dtype == torch.uint8
        images = images.float()
        if was_uint8 or images.detach().amax() > 2.0:
            images = images / 255.0
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.normalize_images:
            images = (images - self.image_mean) / self.image_std
        return images

    def _forward_backbone(self, images: Tensor) -> Tensor:
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(images)
        else:
            out = self.backbone(images)
        if isinstance(out, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooled_output"):
                if key in out:
                    out = out[key]
                    break
            else:
                if "x_norm_patchtokens" in out:
                    out = out["x_norm_patchtokens"].mean(dim=1)
                else:
                    raise ValueError(
                        f"Unsupported DINOv2 output keys: {sorted(out.keys())}"
                    )
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]
        if out.ndim != 2:
            raise ValueError(f"DINOv2 feature must be [B, D], got {tuple(out.shape)}")
        return out


def build_image_encoder(cfg: ModeNetConfig) -> nn.Module:
    if cfg.image_encoder == "small_cnn":
        return SmallImageEncoder(cfg.image_channels, cfg.d_model)
    if cfg.image_encoder.startswith("dinov2_"):
        return DinoV2ImageEncoder(
            model_name=cfg.image_encoder,
            d_model=cfg.d_model,
            feature_dim=cfg.image_feature_dim,
            repo=cfg.dino_repo,
            source=cfg.dino_source,
            pretrained=cfg.dino_pretrained,
            freeze=cfg.freeze_image_encoder,
            image_size=cfg.dino_image_size,
            normalize_images=cfg.normalize_images,
        )
    raise ValueError(
        "image_encoder must be 'small_cnn' or a DINOv2 torch.hub model name "
        "such as 'dinov2_vits14'"
    )


class ModeNet(nn.Module):
    """Transformer for online MAAP mode/allocation prediction.

    Inputs are histories of language/task context, global and wrist images,
    proprioception, previous low-level actions, and previous allocation. The
    output is one continuous allocation value per arm:

        alpha_t^i in [0, 1]

    The values are independent sigmoids by default, so multiple arms can be
    highly active during handover/contact phases.
    """

    def __init__(self, cfg: ModeNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.max_agents <= 0:
            raise ValueError("max_agents must be positive")
        if cfg.image_encoder == "small_cnn" and cfg.d_model % 8 != 0:
            raise ValueError("d_model must be divisible by 8 for GroupNorm")
        if cfg.output_activation not in {"sigmoid", "none"}:
            raise ValueError("output_activation must be 'sigmoid' or 'none'")

        self.image_encoder = build_image_encoder(cfg)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.q_dim + cfg.action_dim + 1, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.arm_query = nn.Parameter(torch.zeros(1, cfg.max_agents, cfg.d_model))
        self.time_emb = nn.Embedding(cfg.max_history, cfg.d_model)
        self.arm_emb = nn.Embedding(cfg.max_agents, cfg.d_model)
        self.type_emb = nn.Embedding(6, cfg.d_model)

        self.task_emb = (
            nn.Embedding(cfg.num_tasks, cfg.d_model)
            if cfg.num_tasks is not None
            else None
        )
        self.language_token_emb = (
            nn.Embedding(cfg.vocab_size, cfg.d_model)
            if cfg.vocab_size is not None
            else None
        )
        self.language_proj = (
            nn.Linear(cfg.language_dim, cfg.d_model)
            if cfg.language_dim is not None
            else None
        )
        self.language_pos_emb = nn.Embedding(cfg.max_language_tokens, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.arm_query, std=0.02)

    def forward(
        self,
        *,
        wrist_images: Tensor,
        q: Tensor,
        prev_actions: Tensor,
        prev_alpha: Tensor,
        global_images: Optional[Tensor] = None,
        language_tokens: Optional[Tensor] = None,
        language_emb: Optional[Tensor] = None,
        task_id: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        return_logits: bool = True,
    ) -> dict[str, Tensor]:
        """Predict current per-arm allocation.

        Args:
            wrist_images: ``[B, T, N, C, H, W]`` wrist-camera history.
            q: ``[B, T, N, q_dim]`` proprioception history.
            prev_actions: ``[B, T, N, action_dim]`` or ``[B, T-1, N, action_dim]``.
            prev_alpha: ``[B, N]`` or ``[B, T, N]`` previous allocation history.
            global_images: optional ``[B, T, C, H, W]`` global image history.
            language_tokens: optional ``[B, L]`` token ids when ``vocab_size`` is set.
            language_emb: optional ``[B, L, language_dim]`` external VLM embeddings.
            task_id: optional ``[B]`` task ids when ``num_tasks`` is set.
            agent_mask: optional bool ``[B, N]``. False agents are ignored and output 0.
            return_logits: include raw logits in the returned dict.

        Returns:
            ``{"alpha": [B, N]}``, plus ``"logits"`` when requested.
        """
        self._validate_core_inputs(wrist_images, q, prev_actions, prev_alpha)
        batch_size, history, num_agents = wrist_images.shape[:3]
        device = wrist_images.device

        if history > self.cfg.max_history:
            raise ValueError(
                f"history length {history} exceeds max_history={self.cfg.max_history}"
            )
        if num_agents > self.cfg.max_agents:
            raise ValueError(
                f"num_agents {num_agents} exceeds max_agents={self.cfg.max_agents}"
            )

        if agent_mask is None:
            agent_mask = torch.ones(
                batch_size, num_agents, dtype=torch.bool, device=device
            )
        else:
            agent_mask = agent_mask.to(device=device, dtype=torch.bool)
            if agent_mask.shape != (batch_size, num_agents):
                raise ValueError(
                    f"agent_mask must have shape {(batch_size, num_agents)}, "
                    f"got {tuple(agent_mask.shape)}"
                )

        tokens: list[Tensor] = []
        padding_masks: list[Tensor] = []

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls + self.type_emb_id(0, device))
        padding_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))

        self._append_language_tokens(
            tokens=tokens,
            padding_masks=padding_masks,
            batch_size=batch_size,
            device=device,
            language_tokens=language_tokens,
            language_emb=language_emb,
            task_id=task_id,
        )

        if global_images is not None:
            tokens.append(self._encode_global_tokens(global_images, history))
            padding_masks.append(torch.zeros(batch_size, history, dtype=torch.bool, device=device))

        arm_time_tokens = self._encode_arm_time_tokens(
            wrist_images=wrist_images,
            q=q,
            prev_actions=self._align_actions(prev_actions, history),
            prev_alpha=self._align_alpha(prev_alpha, history),
        )
        tokens.append(arm_time_tokens)
        padding_masks.append(
            (~agent_mask).unsqueeze(1).expand(batch_size, history, num_agents).reshape(
                batch_size, history * num_agents
            )
        )

        query_tokens = self._make_arm_queries(batch_size, num_agents, device)
        tokens.append(query_tokens)
        padding_masks.append(~agent_mask)

        x = torch.cat(tokens, dim=1)
        padding_mask = torch.cat(padding_masks, dim=1)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        query_out = x[:, -num_agents:]
        logits = self.alpha_head(query_out).squeeze(-1)
        logits = logits.masked_fill(~agent_mask, -30.0)
        alpha = torch.sigmoid(logits) if self.cfg.output_activation == "sigmoid" else logits
        alpha = alpha.masked_fill(~agent_mask, 0.0)

        out = {"alpha": alpha}
        if return_logits:
            out["logits"] = logits
        return out

    def type_emb_id(self, idx: int, device: torch.device) -> Tensor:
        return self.type_emb(torch.tensor(idx, device=device)).view(1, 1, -1)

    def _append_language_tokens(
        self,
        *,
        tokens: list[Tensor],
        padding_masks: list[Tensor],
        batch_size: int,
        device: torch.device,
        language_tokens: Optional[Tensor],
        language_emb: Optional[Tensor],
        task_id: Optional[Tensor],
    ) -> None:
        if task_id is not None:
            if self.task_emb is None:
                raise ValueError("task_id was provided, but num_tasks is not configured")
            task_token = self.task_emb(task_id.to(device)).unsqueeze(1)
            tokens.append(task_token + self.type_emb_id(1, device))
            padding_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))

        if language_tokens is not None:
            if self.language_token_emb is None:
                raise ValueError(
                    "language_tokens were provided, but vocab_size is not configured"
                )
            lang = self.language_token_emb(language_tokens.to(device))
        elif language_emb is not None:
            if self.language_proj is None:
                raise ValueError(
                    "language_emb was provided, but language_dim is not configured"
                )
            lang = self.language_proj(language_emb.to(device))
        else:
            return

        if lang.ndim != 3 or lang.shape[0] != batch_size:
            raise ValueError("language input must have shape [B, L, D]")
        lang_len = lang.shape[1]
        if lang_len > self.cfg.max_language_tokens:
            raise ValueError(
                f"language length {lang_len} exceeds "
                f"max_language_tokens={self.cfg.max_language_tokens}"
            )
        pos = self.language_pos_emb(torch.arange(lang_len, device=device)).unsqueeze(0)
        lang = lang + pos + self.type_emb_id(2, device)
        tokens.append(lang)
        padding_masks.append(torch.zeros(batch_size, lang_len, dtype=torch.bool, device=device))

    def _encode_global_tokens(self, global_images: Tensor, history: int) -> Tensor:
        if global_images.ndim != 5:
            raise ValueError("global_images must have shape [B, T, C, H, W]")
        batch_size, global_history = global_images.shape[:2]
        if global_history != history:
            raise ValueError("global_images history length must match wrist_images")
        encoded = self.image_encoder(global_images.reshape(-1, *global_images.shape[2:]))
        encoded = encoded.reshape(batch_size, history, -1)
        time_ids = torch.arange(history, device=global_images.device)
        return encoded + self.time_emb(time_ids).unsqueeze(0) + self.type_emb_id(
            3, global_images.device
        )

    def _encode_arm_time_tokens(
        self,
        *,
        wrist_images: Tensor,
        q: Tensor,
        prev_actions: Tensor,
        prev_alpha: Tensor,
    ) -> Tensor:
        batch_size, history, num_agents = wrist_images.shape[:3]
        image_shape = wrist_images.shape[3:]
        image_feat = self.image_encoder(wrist_images.reshape(-1, *image_shape))
        image_feat = image_feat.reshape(batch_size, history, num_agents, -1)

        state = torch.cat([q, prev_actions, prev_alpha.unsqueeze(-1)], dim=-1)
        state_feat = self.state_proj(state)
        token = image_feat + state_feat

        time_ids = torch.arange(history, device=wrist_images.device)
        arm_ids = torch.arange(num_agents, device=wrist_images.device)
        token = token + self.time_emb(time_ids).view(1, history, 1, -1)
        token = token + self.arm_emb(arm_ids).view(1, 1, num_agents, -1)
        token = token + self.type_emb_id(4, wrist_images.device).view(1, 1, 1, -1)
        return token.reshape(batch_size, history * num_agents, -1)

    def _make_arm_queries(
        self, batch_size: int, num_agents: int, device: torch.device
    ) -> Tensor:
        arm_ids = torch.arange(num_agents, device=device)
        queries = self.arm_query[:, :num_agents].expand(batch_size, -1, -1)
        queries = queries + self.arm_emb(arm_ids).unsqueeze(0)
        return queries + self.type_emb_id(5, device)

    def _align_actions(self, prev_actions: Tensor, history: int) -> Tensor:
        if prev_actions.shape[1] == history:
            return prev_actions
        if prev_actions.shape[1] != history - 1:
            raise ValueError(
                "prev_actions must have T or T-1 history dimension relative to wrist_images"
            )
        pad = torch.zeros_like(prev_actions[:, :1])
        return torch.cat([pad, prev_actions], dim=1)

    def _align_alpha(self, prev_alpha: Tensor, history: int) -> Tensor:
        if prev_alpha.ndim == 2:
            return prev_alpha.unsqueeze(1).expand(-1, history, -1)
        if prev_alpha.ndim == 3 and prev_alpha.shape[1] == history:
            return prev_alpha
        raise ValueError("prev_alpha must have shape [B, N] or [B, T, N]")

    def _validate_core_inputs(
        self, wrist_images: Tensor, q: Tensor, prev_actions: Tensor, prev_alpha: Tensor
    ) -> None:
        if wrist_images.ndim != 6:
            raise ValueError("wrist_images must have shape [B, T, N, C, H, W]")
        batch_size, history, num_agents = wrist_images.shape[:3]
        expected_q = (batch_size, history, num_agents, self.cfg.q_dim)
        if tuple(q.shape) != expected_q:
            raise ValueError(f"q must have shape {expected_q}, got {tuple(q.shape)}")
        if prev_actions.ndim != 4:
            raise ValueError("prev_actions must have shape [B, T or T-1, N, action_dim]")
        if prev_actions.shape[0] != batch_size or prev_actions.shape[2:] != (
            num_agents,
            self.cfg.action_dim,
        ):
            raise ValueError("prev_actions batch/agent/action dimensions do not match")
        if prev_alpha.ndim not in {2, 3}:
            raise ValueError("prev_alpha must have shape [B, N] or [B, T, N]")
        if prev_alpha.shape[0] != batch_size or prev_alpha.shape[-1] != num_agents:
            raise ValueError("prev_alpha batch/agent dimensions do not match")


Modenet = ModeNet
