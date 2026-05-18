# Multi Agent Active Perception (MAAP)

## Four table tasks

| Task | Config | Description |
|------|--------|-------------|
| **PlaceCubeOnCart** | `configs/table/place_cube_on_cart.yaml` | Two robots: one observes, one places cube on cart. |
| **TwoRobotsStackCubeActive** | `configs/table/two_robots_stack_cube_active.yaml` | Two robots stack two cubes with active perception. |
| **PickMeatFromPot** | `configs/table/pick_meat_from_pot.yaml` | Two robots: one observes, one picks meat from pot to goal. |
| **PickMeatFromMicrowave** | `configs/table/pick_meat_from_microwave.yaml` | Three robots: observe, open door, pick meat to goal. |

---

## Test run (single trajectory with GUI)

From `robofactory/`:

```bash
python script/run_task.py configs/table/place_cube_on_cart.yaml
```

Other tasks:

```bash
python script/run_task.py configs/table/two_robots_stack_cube_active.yaml
python script/run_task.py configs/table/pick_meat_from_pot.yaml
python script/run_task.py configs/table/pick_meat_from_microwave.yaml
```

---

## Generate data (motion-planning demos)

From `robofactory/`:

```bash
python script/generate_data.py --config configs/table/place_cube_on_cart.yaml --num 1 --save-video
```

- `--num N`: number of successful trajectories to record.
- `--save-video`: save videos of each trajectory.

Example for more trajectories:

```bash
python script/generate_data.py --config configs/table/place_cube_on_cart.yaml --num 100 --save-video
```

Same pattern for the other three tasks (replace the config path).

**Generate data for DP3 (pointcloud):**

From `robofactory/`, use `generate_data_pointcloud.py` to record pointcloud observations (requires panda_wristcam):

```bash
python script/generate_data_pointcloud.py --config configs/table/place_cube_on_cart.yaml --num 100 --save-video
```

- `--resume`: resume from existing data (断点继续).
- `--resume-from <path>`: path to existing `.json` to continue from.

Same pattern for the other tasks (replace the config path).

---

## Convert data for Policy-Lightning

From `Policy-Lightning-main/`, convert robofactory `.h5` demos into the format expected by Policy-Lightning (per-agent `head_cam_*` and `action_*`):

```bash
python script/image/extract.py --dataset_path={dataset_path} --output_path={output_path} --load_num 50 --agent_num {agent_num}
```

- `--dataset_path`: path to the source `.h5` from robofactory (e.g. `../robofactory/demos/PlaceCubeOnCart-rf/motionplanning/20250101_120000.h5`).
- `--output_path`: path for the output `.h5` (e.g. `data/place_cube_on_cart.h5`).
- `--load_num`: number of trajectories to load (use `50` or `-1` for all).
- `--agent_num`: number of agents (`2` for PlaceCubeOnCart, TwoRobotsStackCubeActive, PickMeatFromPot; `3` for PickMeatFromMicrowave).

Example (2 agents, 50 trajs):

```bash
python script/image/extract.py --dataset_path=../robofactory/demos/PlaceCubeOnCart-rf/motionplanning/20250101_120000.h5 --output_path=data/place_cube_on_cart.h5 --load_num 50 --agent_num 2
```

**Convert pointcloud data for DP3:**

From `Policy-Lightning-main/`, after generating robofactory demos with pointcloud mode:

```bash
python script/pointcloud/extract.py \
  --dataset_path ../robofactory/demos/PlaceCubeOnCart-rf/motionplanning/YYYYMMDD_HHMMSS.h5 \
  --output_path data/place_cube_on_cart_pointcloud.h5 \
  --load_num -1 --agent_num 2 --point_num 512
```

- `--dataset_path`: path to the source `.h5` from robofactory (use the merged `.h5` next to the `.json`).
- `--output_path`: path for the output `.h5` (e.g. `data/place_cube_on_cart_pointcloud.h5`).
- `--load_num`: number of trajectories (`-1` for all).
- `--agent_num`: number of agents (`2` for PlaceCubeOnCart, PickMeatFromPot; `3` for PickMeatFromMicrowave).
- `--point_num`: points per cloud (default `512`).

---

## Training (Policy-Lightning)

From `Policy-Lightning-main/`.

Data is expected under `Policy-Lightning-main/data/` (e.g. `place_cube_on_cart.h5`) after converting/exporting the `robofactory` demos.

**Local DP2 (each agent sees all agents’ cameras):**

```bash
python workspace.py --config-name local_dp2 task=place_cube_on_cart_local
python workspace.py --config-name local_dp2 task=2a_two_robots_stack_cube_active_local
python workspace.py --config-name local_dp2 task=pick_meat_from_pot_local
python workspace.py --config-name local_dp2 task=pick_meat_from_microwave_local
```

**Local DP2 single-cam (all agents use one camera agent’s image):**

```bash
python workspace.py --config-name local_dp2_single_cam task=place_cube_on_cart_local_single_cam
python workspace.py --config-name local_dp2_single_cam task=2a_two_robots_stack_cube_active_local_single_cam
python workspace.py --config-name local_dp2_single_cam task=pick_meat_from_pot_local_single_cam
python workspace.py --config-name local_dp2_single_cam task=pick_meat_from_microwave_local_single_cam
```

**Local DP3 (each agent uses own pointcloud):**


```bash
python workspace.py --config-name local_dp3 task=place_cube_on_cart_3d_local
python workspace.py --config-name local_dp3 task=2a_two_robots_stack_cube_active_3d_local
python workspace.py --config-name local_dp3 task=pick_meat_from_pot_3d_local
python workspace.py --config-name local_dp3 task=pick_meat_from_microwave_3d_local
```

**Local DP3 single-cam (all agents use one pointcloud source):**

```bash
python workspace.py --config-name local_dp3_single_cam task=place_cube_on_cart_3d_local_single_cam
python workspace.py --config-name local_dp3_single_cam task=2a_two_robots_stack_cube_active_3d_local_single_cam
python workspace.py --config-name local_dp3_single_cam task=pick_meat_from_pot_3d_local_single_cam
python workspace.py --config-name local_dp3_single_cam task=pick_meat_from_microwave_3d_local_single_cam
```

**Global DP3 (global policy, each agent uses own pointcloud):**

```bash
python workspace.py --config-name global_dp3 task=place_cube_on_cart_3d_global
python workspace.py --config-name global_dp3 task=2a_two_robots_stack_cube_active_3d_global
python workspace.py --config-name global_dp3 task=pick_meat_from_pot_3d_global
python workspace.py --config-name global_dp3 task=pick_meat_from_microwave_3d_global
```

**Global DP3 single-cam (global policy, all agents use one pointcloud source):**

```bash
python workspace.py --config-name global_dp3_single_cam task=place_cube_on_cart_3d_global_single_cam
python workspace.py --config-name global_dp3_single_cam task=2a_two_robots_stack_cube_active_3d_global_single_cam
python workspace.py --config-name global_dp3_single_cam task=pick_meat_from_pot_3d_global_single_cam
python workspace.py --config-name global_dp3_single_cam task=pick_meat_from_microwave_3d_global_single_cam
```
