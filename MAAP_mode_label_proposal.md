# MAAP Per-Arm Action/Perception Label Proposal

## 核心定义

这次不要把 label 理解成“这一帧整个 task 是 action 还是 perception”。正确的定义是：

```text
z_t^i in {0, 1}

i: 第 i 个机械臂
t: 当前 action/frame
0: perception / watch / observe
1: doing action / active manipulation
```

也就是说，每一帧、每一个机械臂都有一个二值 label。一个三臂任务在第 t 帧的 label 不是一个数，而是类似：

```text
[arm0, arm1, arm2] = [0, 0, 1]
```

含义是：arm2 正在做真正改变世界状态的操作，arm0 和 arm1 在看、提供视觉上下文，或者只是保持姿态。

## Label 语义

`1 = action` 只给当前真正负责物体交互的臂：

- grasp object
- close gripper to hold object/handle
- pull/open articulated object
- lift/move/place object
- release object as part of manipulation

`0 = perception` 给所有非操作臂，以及所有只为观察/视角准备服务的运动：

- 移到观察位
- wrist camera 对准 task-relevant region
- 等待另一个臂操作
- 保持姿态提供多视角观测
- setup motion，如果它没有直接接触或改变物体状态

这个定义比“移动就是 action”更严格。机械臂移动到观察位时，虽然低层 action 非零，但语义上仍然是 perception。

## 四个 MAAP 任务的标注规则

### PlaceCubeOnCart

实际 solution 中：

- `panda_wristcam-0`: 移到 top-down/cart/cube 观察位，全程 label `0`
- `panda_wristcam-1`: pick cube、move to cart、place cube，操作段 label `1`

最新生成数据：

```text
PlaceCubeOnCart-rf / 20260520_163324.h5
panda_wristcam-0: 0-350 全部 0
panda_wristcam-1: 0-55 为 0, 56-350 为 1
```

### PickMeatFromPot

实际 solution 中：

- `panda_wristcam-0`: 移到 pot/meat 观察位，全程 label `0`
- `panda_wristcam-1`: pick meat、lift、place 到 goal region，操作段 label `1`

最新生成数据：

```text
PickMeatFromPot-rf / 20260520_163352.h5
panda_wristcam-0: 0-252 全部 0
panda_wristcam-1: 0-40 为 0, 41-252 为 1
```

### TwoRobotsStackCubeActive

实际 solution 中 manipulator 是动态选出来的：

- 离两个 cube centroid 更近的臂是 manipulator
- 另一个臂是 observer，先移到 viewing pose，然后全程 label `0`
- manipulator pick/place cubeA，再 pick/stack cubeB，操作段 label `1`

最新这一条里实际选到：

```text
TwoRobotsStackCubeActive-rf / 20260520_163418.h5
panda_wristcam-0: 0-413 全部 0
panda_wristcam-1: 0-77 为 0, 78-413 为 1
```

### PickMeatFromMicrowave

这个任务最容易误解，必须按 solution 里的实际 `move_id` 看，而不是按旧注释猜。

最新理解是三段：

1. setup/perception：所有臂都是 `0`
2. 开 microwave 门：`panda_wristcam-2 = 1`，另外两个臂 `0`
3. 拿 meat：`panda_wristcam-1 = 1`，另外两个臂 `0`

最新生成数据：

```text
PickMeatFromMicrowave-rf / 20260520_162552.h5
panda_wristcam-0: 0-598 全部 0
panda_wristcam-2: 0-97 为 0, 98-330 为 1, 331-598 为 0
panda_wristcam-1: 0-408 为 0, 409-598 为 1
```

## 数据格式

H5 中每个 trajectory 保存 per-agent labels：

```text
traj_i/mode_labels/panda_wristcam-0
traj_i/mode_labels/panda_wristcam-1
traj_i/mode_labels/panda_wristcam-2   # 三臂任务才有
```

每个 dataset 都和 action 对齐：

```text
len(mode_labels[panda_wristcam-k]) == len(actions[panda_wristcam-k])
```

这样 downstream 不需要猜当前 action 属于哪个臂，直接读取对应 agent 的 label。

## 怎么学习这个 label

第一阶段可以把 label 当成 planner 生成的 pseudo-label，训练一个 per-arm ModeNet：

```text
z_hat_t^i = ModeNet(o_{t-k:t}, q_{t-k:t}^i, a_{t-k:t-1}^i, task)
```

输出不是单个全局 label，而是每个 arm 一个二分类：

```text
Z_hat_t = [z_hat_t^0, z_hat_t^1, ..., z_hat_t^N]
```

loss：

```text
L_mode = sum_i CE(z_hat_t^i, z_t^i)
```

输入建议包括：

- 当前和历史图像，多视角 wrist/global 都可以
- 每个机械臂 proprioception
- 上几步动作
- task id / language instruction
- 可选：agent id embedding，让网络知道当前预测的是哪个 arm

这个 label 的难点不是 frame-level visual classification，而是 role/phase classification：模型要学会“现在这个臂是在看，还是轮到它动手改变物体状态”。

## 怎么给 VLA 用

VLA 有两种直接用法。

第一种是作为 action policy 的条件：

```text
pi(a_t^i | obs, q^i, instruction, z_t^i)
```

训练时用 ground-truth label；推理时用 ModeNet 预测的 `z_hat_t^i`。这样 VLA 会显式知道当前这个 arm 应该执行观察动作还是操作动作。

第二种是作为辅助头一起学：

```text
z_hat_t^i, a_hat_t^i = VLA(obs, q, instruction, agent_id=i)
L = L_action + lambda * L_mode
```

推荐先做两阶段：

1. 先训练 ModeNet，看 per-arm label 是否能预测准
2. 再把 `z_hat_t^i` 接到 VLA 里做 mode-conditioned imitation

两阶段更容易 debug：如果失败，可以分清是 mode 预测错，还是动作模型在正确 mode 下也做不好。

## 怎么给 WAM 用

WAM 里 label 更像一个 high-level dynamics switch。对每个 arm：

```text
WAM(o_t, q_t, a_t, Z_t) -> o_{t+1:t+H}, q_{t+1:t+H}, object_state_{t+1:t+H}
```

其中：

```text
Z_t = [z_t^0, z_t^1, ..., z_t^N]
```

直觉是：

- 如果某个 arm 是 `0`，它的动作主要改变视角/相机观测，不应该大幅改变 object state
- 如果某个 arm 是 `1`，它的动作可能改变 contact、object pose、articulation state

所以 WAM 不应该只看低层 action，而应该知道“这个 action 在语义上是观察还是操作”。这能减少 dynamics ambiguity。

规划时也可以把 `Z_t` 当成离散高层变量搜索：

```text
[0,0,1] -> [0,0,1] -> [0,1,0]
```

例如 microwave：

```text
all perception/setup
arm2 action: open door
arm1 action: pick meat
```

WAM 预测每种 mode assignment 的后果，VLA 负责在给定 mode 下生成低层动作。

## 推荐实验

先做最小闭环：

1. 用四个 MAAP task 生成 per-arm labeled demos
2. 用 label viewer 检查每条轨迹的 frame-label alignment
3. 训练 per-arm ModeNet
4. 做 VLA ablation：
   - no label
   - ground-truth label condition
   - predicted label condition
   - auxiliary label head only
5. 做 WAM ablation：
   - no label conditioning
   - global label conditioning
   - per-arm label conditioning

关键指标：

- task success rate
- per-arm label accuracy / F1
- transition frame accuracy
- 在 mode 预测错误时，VLA/WAM 是否出现可解释失败
- WAM 是否能更好地区分“看导致观测变化”和“操作导致物体状态变化”

## 当前实现状态

已经完成：

- 采集时写入 per-arm `mode_labels`
- motion planner 支持对指定 `agent_ids` 设置 label
- 四个 README task 各重新生成了一条成功数据
- label viewer 支持同时看视频帧和每个 arm 的 label timeline

当前 viewer：

```text
robofactory/label_viewer/index.html
```

当前四条数据都在：

```text
robofactory/demos/*/motionplanning/*.h5
```
