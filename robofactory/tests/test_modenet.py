import torch

from modenet import ModeNet, ModeNetConfig


def test_modenet_forward_shapes():
    cfg = ModeNetConfig(
        max_agents=3,
        q_dim=8,
        action_dim=8,
        image_encoder="small_cnn",
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_history=4,
        vocab_size=32,
        num_tasks=8,
    )
    model = ModeNet(cfg)

    batch_size = 2
    history = 4
    agents = 2
    out = model(
        wrist_images=torch.randn(batch_size, history, agents, 3, 64, 64),
        global_images=torch.randn(batch_size, history, 3, 64, 64),
        q=torch.randn(batch_size, history, agents, 8),
        prev_actions=torch.randn(batch_size, history - 1, agents, 8),
        prev_alpha=torch.rand(batch_size, agents),
        language_tokens=torch.randint(0, 32, (batch_size, 6)),
        task_id=torch.randint(0, 8, (batch_size,)),
    )

    assert out["alpha"].shape == (batch_size, agents)
    assert out["logits"].shape == (batch_size, agents)
    assert torch.all(out["alpha"] >= 0)
    assert torch.all(out["alpha"] <= 1)


def test_modenet_agent_mask_zeroes_inactive_agents():
    cfg = ModeNetConfig(
        max_agents=3,
        q_dim=4,
        action_dim=5,
        image_encoder="small_cnn",
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_history=3,
        language_dim=16,
    )
    model = ModeNet(cfg)
    agent_mask = torch.tensor([[True, False, True]])
    out = model(
        wrist_images=torch.randn(1, 3, 3, 3, 32, 32),
        q=torch.randn(1, 3, 3, 4),
        prev_actions=torch.randn(1, 3, 3, 5),
        prev_alpha=torch.rand(1, 3, 3),
        language_emb=torch.randn(1, 5, 16),
        agent_mask=agent_mask,
    )

    assert out["alpha"].shape == (1, 3)
    assert out["alpha"][0, 1].item() == 0.0
