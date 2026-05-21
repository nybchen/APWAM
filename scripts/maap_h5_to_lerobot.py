"""
Convert MAAP (robofactory) HDF5 trajectories into a LeRobot dataset that
LingBot-VA can post-train on.

Source h5 layout (from RecordEpisodeMA, see robofactory/utils/wrappers/record.py):

  traj_{i}/
    obs/
      sensor_data/
        hand_camera_0/rgb     (T, H, W, 3) uint8
        hand_camera_1/rgb     (T, H, W, 3) uint8
        head_camera_global/rgb(T, H, W, 3) uint8
        <cam>/sensor_param/...
      agent/qpos, qvel, ...
      extra/...
    actions/
      panda_wristcam-0        (T-1, 8)   # 7 joint deltas + 1 gripper
      panda_wristcam-1        (T-1, 8)
    terminated, truncated, success, fail

Target LingBot-VA layout (see lingbot-va/README.md "Custom Dataset Preparation"):

  <out>/
    meta/
      info.json
      episodes.jsonl         (with action_config per episode)
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.cam_high/episode_000000.mp4
    videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
    videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4

The downstream VAE-latent extraction (Wan2.2) is a separate step run in the
lingbot-va env; this script only produces the LeRobot side.

USAGE
-----
    python scripts/maap_h5_to_lerobot.py \
        --h5 robofactory/demos/TwoRobotsStackCubeActive-rf/motionplanning/<ts>.h5 \
        --out ~/datasets/maap_two_robots_stack_active_lerobot \
        --task-text "Two robots stack cubeB on cubeA at the goal region" \
        --fps 10

The script is intentionally dependency-light on the robofactory side: it only
reads h5 with h5py. It DOES import lerobot, so run it inside the lingbot-va env
(which has `lerobot==0.3.3`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Mapping from MAAP camera uids to LingBot-VA expected camera keys.
# LingBot-VA's robotwin config hard-codes these three keys
# (see lingbot-va/wan_va/configs/va_robotwin_cfg.py:19).
# ---------------------------------------------------------------------------
CAM_MAP: dict[str, str] = {
    "head_camera_global": "observation.images.cam_high",
    "hand_camera_0": "observation.images.cam_left_wrist",
    "hand_camera_1": "observation.images.cam_right_wrist",
}

# MAAP robot uids in the order their actions concatenate into the 14-dim block.
ROBOT_UIDS_2A: tuple[str, ...] = ("panda_wristcam-0", "panda_wristcam-1")


def maap_action_to_lingbot30(
    act_left: np.ndarray,  # (T, 8) — 7 joint + 1 gripper for agent 0
    act_right: np.ndarray,  # (T, 8) — 7 joint + 1 gripper for agent 1
) -> np.ndarray:
    """Pack two 8-dim Panda actions into LingBot-VA's 30-dim slot.

    LingBot expects (per va_robotwin_cfg.py):
        [0:7]   left EEF (xyz + quat) — MAAP doesn't have EEF target action, pad 0
        [7:14]  right EEF             — pad 0
        [14:21] left joints (7)       — from act_left[:, :7]
        [21:28] right joints (7)      — from act_right[:, :7]
        [28]    left gripper          — act_left[:, 7]
        [29]    right gripper         — act_right[:, 7]

    Channels not used will be ignored by `used_action_channel_ids`; we only fill
    the joints + grippers, and let the EEF block stay zero. norm_stat will be
    recomputed at training time from this dataset's actual range.
    """
    assert act_left.shape == act_right.shape, (act_left.shape, act_right.shape)
    assert act_left.shape[1] == 8, f"expected (T,8), got {act_left.shape}"
    T = act_left.shape[0]
    out = np.zeros((T, 30), dtype=np.float32)
    out[:, 14:21] = act_left[:, :7]
    out[:, 21:28] = act_right[:, :7]
    out[:, 28] = act_left[:, 7]
    out[:, 29] = act_right[:, 7]
    return out


def iter_episodes(h5_path: Path):
    """Yield (episode_index, dict(cam_key -> (T,H,W,3) uint8), action (T,30)) per traj."""
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda k: int(k.split("_")[1]),
        )
        for ep_idx, traj_key in enumerate(traj_keys):
            g = f[traj_key]

            # --- images ---
            sensor = g["obs/sensor_data"]
            imgs: dict[str, np.ndarray] = {}
            for src_cam, dst_key in CAM_MAP.items():
                if src_cam not in sensor:
                    raise KeyError(
                        f"{traj_key}: expected camera '{src_cam}' in obs/sensor_data, "
                        f"found {list(sensor.keys())}"
                    )
                rgb = sensor[f"{src_cam}/rgb"][...]  # (T, H, W, 3) uint8
                imgs[dst_key] = rgb

            # --- actions ---
            # MAAP stores actions starting from step 1 (length T-1).
            # Pad once at the front with zeros so action length == image length T.
            actions_grp = g["actions"]
            act_left = actions_grp[ROBOT_UIDS_2A[0]][...].astype(np.float32)
            act_right = actions_grp[ROBOT_UIDS_2A[1]][...].astype(np.float32)
            action14 = np.concatenate([act_left, act_right], axis=0)  # for shape check
            del action14
            action30_short = maap_action_to_lingbot30(act_left, act_right)  # (T-1, 30)
            pad = np.zeros((1, 30), dtype=np.float32)
            action30 = np.concatenate([pad, action30_short], axis=0)  # (T, 30)

            # sanity check: align with first image's T
            T_img = next(iter(imgs.values())).shape[0]
            if action30.shape[0] != T_img:
                # If you see this, the assumption above (front-pad) is wrong for this run;
                # inspect the h5 manually and adjust.
                raise ValueError(
                    f"{traj_key}: action length {action30.shape[0]} != image length {T_img}"
                )

            yield ep_idx, imgs, action30


def build_features_dict(image_height: int, image_width: int) -> dict:
    """Schema for LeRobotDataset.create. Must match the cam_keys downstream
    expects (see CAM_MAP) and the 30-dim action vector."""
    feat: dict = {
        "action": {
            "dtype": "float32",
            "shape": (30,),
            "names": [f"a{i}" for i in range(30)],
        },
    }
    for dst_key in CAM_MAP.values():
        feat[dst_key] = {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        }
    return feat


def patch_episodes_jsonl_with_action_config(
    out_dir: Path, action_text: str
) -> None:
    """Append `action_config` field to every line in meta/episodes.jsonl.

    LeRobotDataset writes episodes.jsonl with episode_index / tasks / length;
    LingBot-VA additionally needs an `action_config` list. We add ONE segment
    per episode covering [0, length) with the same NL description.
    """
    jsonl_path = out_dir / "meta" / "episodes.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    lines_out: list[str] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": int(rec["length"]),
                    "action_text": action_text,
                }
            ]
            lines_out.append(json.dumps(rec, ensure_ascii=False))

    with jsonl_path.open("w") as f:
        f.write("\n".join(lines_out) + "\n")


def convert(
    h5_path: Path,
    out_dir: Path,
    task_text: str,
    fps: int,
    repo_id: str,
) -> None:
    # Import lerobot lazily so that --help works even without it installed.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Peek one episode to learn (H, W).
    with h5py.File(h5_path, "r") as f:
        first_traj = next(k for k in f.keys() if k.startswith("traj_"))
        sample_rgb = f[f"{first_traj}/obs/sensor_data/head_camera_global/rgb"]
        H, W = int(sample_rgb.shape[1]), int(sample_rgb.shape[2])

    out_dir.mkdir(parents=True, exist_ok=True)
    features = build_features_dict(H, W)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=out_dir,
        features=features,
        # Disable HF hub push — keep it local-only.
        use_videos=True,
    )

    n_episodes = 0
    for ep_idx, imgs, action30 in iter_episodes(h5_path):
        T = action30.shape[0]
        for t in range(T):
            frame = {dst_key: imgs[dst_key][t] for dst_key in CAM_MAP.values()}
            frame["action"] = action30[t]
            # LeRobot 0.3.3 add_frame signature: add_frame(frame, task=...)
            dataset.add_frame(frame, task=task_text)
        dataset.save_episode()
        n_episodes += 1
        print(f"[ok] episode {ep_idx} written ({T} frames)")

    # Append LingBot-specific action_config field.
    patch_episodes_jsonl_with_action_config(out_dir, task_text)
    print(f"[done] {n_episodes} episodes -> {out_dir}")
    print("Next step: extract Wan2.2 VAE latents under <out>/latents/ "
          "(see lingbot-va/README.md 'Step 3').")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, required=True,
                    help="Path to the merged MAAP trajectory .h5 file.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for the LeRobot dataset.")
    ap.add_argument("--task-text", type=str, required=True,
                    help="Natural-language description used for `task` and `action_text`.")
    ap.add_argument("--fps", type=int, default=10,
                    help="FPS recorded in LeRobot meta. Should match the rate at "
                         "which you intend to sample frames for VAE latents.")
    ap.add_argument("--repo-id", type=str, default="local/maap_two_robots_stack_active",
                    help="LeRobot repo_id. Local-only; not pushed to HF hub.")
    args = ap.parse_args()
    convert(args.h5, args.out, args.task_text, args.fps, args.repo_id)


if __name__ == "__main__":
    main()
