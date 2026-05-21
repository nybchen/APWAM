# 面向可泛化 VLA/WAM 的 Mode-Aware Active Perception

## 摘要

现有 active-perception manipulation 系统通常依赖直接的 behavior cloning：给定图像、机器人本体状态和任务指令，模型直接预测机器人动作。这种方法在训练分布内可以工作，但学到的策略并不知道一条机械臂当前到底是在获取信息，还是在改变世界状态。这个区别在多臂操作中尤其重要：一条机械臂可能正在操作物体，另一条机械臂可能正在移动 wrist camera、等待、或者提供任务相关视角。

本项目研究一个问题：显式的 per-arm action/perception mode label 能否让 active perception 更可泛化。我们从仿真环境开始采集轨迹，第一阶段使用 RoboFactory/MAAP，之后扩展到真机。每条轨迹在每一帧、每一条机械臂上都有一个二值 label：

```text
z_t^i in {0, 1}

i: 第 i 条机械臂
t: 当前 frame/action timestep
0: perception / observation / information gathering
1: action / manipulation / world-changing interaction
```

然后，我们训练模型从视觉观测中预测这些 label，或者从视觉观测加机器人 state 中预测这些 label。最后，我们评估这些 label 是否能帮助 Vision-Language-Action models (VLA) 和 World Action Models (WAM) 学到能够跨任务、跨 layout、跨遮挡模式、跨机器人角色、跨真实环境迁移的 active perception。

核心研究问题是：

```text
显式的 per-arm action/perception label 能否把 active perception
从 behavior cloning 轨迹里隐含的行为，变成一种可学习、可解释、可泛化的能力？
```

## 背景与动机

MAAP 为这个项目提供了一个很好的起点。MAAP IROS 论文提出，多智能体操作天然会产生 distributed active perception：每条操作臂都有 wrist camera，这些相机可以在机械臂执行任务的同时提供任务相关视角。MAAP 的实验表明，distributed multi-wrist perception 优于固定 global camera，也经常优于 single active wrist view，尤其是在严重遮挡和多阶段任务中，例如打开微波炉并取出物体。

但是，MAAP 仍然主要依赖从专家轨迹中做 imitation learning。控制器接收观测并直接预测动作。demonstration 里面确实存在 temporal role switching：一条机械臂可能负责观察，另一条机械臂负责开门，然后另一条机械臂负责取物体。但这种角色结构并没有被显式表示在学习模型中。模型需要从 behavior cloning 中自己发现每条臂什么时候应该看、什么时候等待、什么时候移动到观察位、什么时候抓取、拉动、抬起或放置。

这带来三个限制：

1. **Active perception 是隐式的。** 策略也许能复现专家动作，但它没有一个显式变量来表示“这条臂当前在获取信息”还是“这条臂当前在改变环境”。
2. **失败难以诊断。** 当策略失败时，很难判断它是选错了角色、用了错误视角，还是低层动作生成错了。
3. **泛化能力受限。** 直接 behavior cloning 容易过拟合到具体任务轨迹，未必能把 role-switching 行为迁移到新的 layout、遮挡、物体位姿或机器人分工中。

本项目的目标就是通过显式建模 temporal role switching 来解决这个问题。

## 核心假设

本项目的假设是：

```text
如果 VLA 和 WAM 模型使用显式的 per-arm action/perception mode label 进行训练，
它们会比只用直接 behavior cloning 的模型学到更可泛化的 active-perception 行为。
```

直觉是，perception 和 manipulation 具有不同的因果效果：

- 在 perception mode 下，机械臂可能移动 wrist camera、减少遮挡、暴露目标物体，或者维持一个有用视角。图像会变化，但任务物体通常不应该发生明显位移。
- 在 action mode 下，机械臂会发生接触、抓取、拉动、推动、抬起、放置或释放。物体状态、articulation state、接触状态或任务进度可能发生变化。

直接 BC 会把这两种 regime 混在同一个 action distribution 里学习。mode-aware model 则可以显式学习任务结构：谁应该看、谁应该操作、什么时候切换角色，以及这些切换如何影响未来观测和状态。

## Label 定义

label 是 per-arm、per-frame 的：

```text
Z_t = [z_t^0, z_t^1, ..., z_t^{N-1}]
```

例如，在一个三臂微波炉任务中：

```text
[0, 0, 1] = arm 2 正在操作；arm 0 和 arm 1 正在观察或等待
[0, 1, 0] = arm 1 正在操作；arm 0 和 arm 2 正在观察或等待
```

`z_t^i = 1` 表示这条机械臂当前负责 world-changing manipulation segment：

- 抓取物体或把 gripper 闭合到物体/把手上
- 拉、推、打开、关闭、抬起、移动、放置或释放物体
- 为了任务进展而保持接触

`z_t^i = 0` 表示这条机械臂当前承担 perception、support 或 non-manipulation role：

- 移动到观察位
- 将 wrist camera 对准任务相关区域
- 等待另一条机械臂完成前置动作
- 保持姿态以提供视觉上下文
- 执行不接触、不改变物体状态的 setup motion

这个定义是语义上的，而不是单纯运动学上的。一条机械臂可以有非零 joint action，但只要这个动作是为了获取视角而不是操作物体，它仍然属于 perception mode。

## 研究目标

本项目有四个目标。

1. **构建带 label 的 active-perception 数据集。** 从仿真和真机轨迹中生成 per-arm、per-frame action/perception label。
2. **学习 mode predictor。** 训练模型从视觉观测中预测每条机械臂的 action/perception mode，或者从视觉观测加机器人 state 中预测。
3. **用 label 改进 VLA。** 测试 mode label 是否能帮助策略学习什么时候观察、什么时候操作，而不只是做直接 behavior cloning。
4. **用 label 改进 WAM。** 测试 mode label 是否能帮助 world/action model 预测 perception action 和 manipulation action 的不同后果。

## 数据采集与 Label 生成

### 仿真数据

第一阶段 testbed 是 RoboFactory/MAAP，因为它已经包含多臂操作任务、wrist camera、planner-generated demonstration 和 active-perception behavior。初始任务包括：

- `PlaceCubeOnCart`
- `PickMeatFromPot`
- `TwoRobotsStackCubeActive`
- `PickMeatFromMicrowave`

这些任务覆盖简单可见性、局部遮挡、严重遮挡、多阶段操作和 temporal role switching。MAAP 论文也提供了有用的 baseline 设置：global fixed view、single active wrist view 和 distributed multi-wrist perception。

仿真 label 可以由多种信号生成：

- scripted planner phase
- active robot 或 `move_id`
- gripper open/close command
- contact state
- object pose 或 articulation state change
- task progress event
- 没有接触的 camera/viewpoint motion

第一版可以使用 planner-derived pseudo-label，因为 scripted expert 本身知道哪条机械臂在开门、取物、放置或观察。后续版本可以把 planner 信号、contact 信号和 object-state change 结合起来，以减少 label noise。

### 真机数据

第二阶段把同样的 label 定义扩展到真机。真机 demonstration 可以来自 teleoperation、scripted collection 或 human-guided active perception。

可用的 label 信号包括：

- robot proprioception 和 end-effector motion
- gripper command 和 gripper width
- force/torque 或 tactile contact
- 来自外部相机或 wrist camera 的 object tracking
- teleoperation logs
- 少量 human annotation，用于 validation subset

目标不是人工标注每一帧，而是使用 weak supervision 和 pseudo-labeling，再训练一个 learned labeler，使其能够跨任务和跨平台迁移。

## 方法

### 1. ModeNet：学习每条机械臂何时观察、何时操作

ModeNet 预测 per-arm mode vector：

```text
Z_hat_t = ModeNet(o_{t-k:t}, q_{t-k:t}, a_{t-k:t-1}, language)
```

其中：

- `o_{t-k:t}` 是 wrist/global image history
- `q_{t-k:t}` 是 robot proprioception
- `a_{t-k:t-1}` 是可选的 previous action history
- `language` 是 task instruction 或 task id
- `Z_hat_t` 包含每条机械臂的一个二分类预测

需要研究两种输入设置：

1. **Vision-only mode prediction。**
   测试仅从视觉上下文中是否能反推出 label。
2. **Vision + state mode prediction。**
   测试 proprioception、gripper state 和 recent motion 是否能提升 role/phase inference。

ModeNet 的评估不能只看 frame accuracy，还要看 transition quality。最难、也最重要的是边界时刻：一条机械臂什么时候从观察变成操作，或者 manipulation responsibility 什么时候从一条臂转移到另一条臂。

### 2. Mode-Conditioned VLA

mode-conditioned VLA 在显式 mode 输入的条件下预测动作：

```text
pi(a_t^i | obs_t, q_t, language, agent_id=i, z_t^i)
```

训练时，策略可以使用 ground-truth 或 pseudo-label mode。推理时，策略使用 ModeNet 的预测：

```text
z_hat_t^i -> VLA -> a_t^i
```

需要比较几种设计：

- **No-label VLA:** 标准 behavior cloning baseline。
- **Ground-truth mode VLA:** 在仿真测试时使用真实 label 的 upper bound。
- **Predicted-mode VLA:** 实际可用的模型，使用 ModeNet 预测 mode。
- **Auxiliary-head VLA:** 同一个网络同时预测 action 和 mode：

```text
L = L_action + lambda * L_mode
```

mode label 可以作为 binary token、per-arm embedding、adapter condition 或 high-level gating variable 注入模型。核心问题不只是 success rate 是否提升，还包括失败是否更可解释：错误的 mode prediction 应该对应可理解的错误，例如观察太久、太早操作，或者把 manipulation phase 分配给了错误的机械臂。

### 3. Mode-Aware WAM

WAM 从当前观测、动作和 mode label 预测未来观测和状态：

```text
WAM(o_t, q_t, a_t, Z_t) -> o_{t+1:t+H}, q_{t+1:t+H}, s_{t+1:t+H}
```

mode vector 告诉 world model 如何解释动作：

- 如果 `z_t^i = 0`，这条臂的运动主要应该影响 camera pose、遮挡关系和信息质量。
- 如果 `z_t^i = 1`，这条臂的运动可能影响 object pose、contact、articulation 和 task progress。

这可以降低 dynamics ambiguity。同样的低层机械臂运动，根据它是在接近观察位还是在执行物体交互，含义可能完全不同。

mode label 也允许在高层搜索 mode sequence：

```text
[0,0,0] -> [0,0,1] -> [0,1,0]
setup/viewing -> door opening -> object retrieval
```

WAM 可以在 VLA 生成具体低层动作之前，先评估某个 role sequence 是否可能暴露关键信息并完成任务。

## 实验计划

### 实验 1：Label 是否可靠？

在 RoboFactory/MAAP 中生成 labeled trajectory，并用 label viewer 检查 frame-level alignment。

指标：

- label consistency across trajectories
- planner label 与 contact/object-change heuristic 的一致性
- transition-frame correctness
- sampled videos 上的 human spot-check accuracy

预期结果：

```text
对于 scripted simulation trajectories，
planner-derived pseudo-label 应该足够可靠，可以训练初始 ModeNet。
```

### 实验 2：ModeNet 能否预测 Action/Perception Mode？

在 labeled trajectories 上训练 ModeNet，并评估泛化能力。

划分方式：

- held-out random seeds
- held-out object poses
- held-out camera/object occlusion patterns
- held-out robot-role assignments
- 后续扩展到 held-out tasks

指标：

- per-arm accuracy
- action mode 的 F1 score
- transition timing error
- perception phase 中误判为 action 的比例
- manipulation phase 中误判为 perception 的比例

关键 ablation：

- vision only
- vision + proprioception
- vision + proprioception + action history
- single-view input
- multi-wrist input
- frozen visual encoder vs. trainable visual encoder

### 实验 3：Label 是否能改进 VLA Active Perception？

训练带 mode label 和不带 mode label 的 VLA/BC policy。

Baselines：

- global fixed view BC
- single active wrist view BC
- 不带 label 的 MAAP-style multi-wrist BC
- 使用 ground-truth label 的 mode-conditioned VLA
- 使用 predicted label 的 mode-conditioned VLA
- auxiliary-mode VLA

指标：

- task success rate
- 新 object pose/layout 下的 success rate
- 更强遮挡下的 success rate
- role-switching failure 的数量和类型
- action smoothness 和不必要 camera motion
- partial occlusion 或第一次尝试失败后的 recovery 能力

预期结果：

```text
mode-conditioned policy 应该在 occlusion-heavy 和 multi-phase tasks 上提升最大，
因为这些任务最依赖 observe/action role switch 的时机。
```

### 实验 4：Label 是否能改进 WAM Prediction 和 Planning？

训练带 mode conditioning 和不带 mode conditioning 的 WAM。

Baselines：

- 不带 mode label 的 WAM
- 带单个 global mode label 的 WAM
- 带 per-arm mode label 的 WAM
- 带 predicted per-arm mode label 的 WAM

指标：

- future image prediction error
- object pose/articulation prediction error
- contact/task-progress prediction accuracy
- predicted information gain 的质量
- mode sequence planning 下的 success rate

预期结果：

```text
per-arm mode label 应该帮助 WAM 区分 viewpoint-changing action
和 world-changing action，从而提升 prediction 和 planning。
```

### 实验 5：Sim-to-Real 和 Real-Only Evaluation

把学到的 labeler 和 mode-conditioned policy 迁移到真机数据。

评估设置：

- 在仿真中训练 ModeNet，在真实 demonstration 上测试
- 用少量真实 pseudo-labeled data fine-tune ModeNet
- 使用仿真 label 训练 VLA/WAM，并评估真实 active-perception behavior
- 将 pseudo-label 与少量 human-labeled validation clips 比较

指标：

- real-data mode prediction accuracy
- 遮挡场景下的 active-perception success
- 对 camera calibration difference 的鲁棒性
- sim-to-real label transfer quality

## 与 MAAP 的关系

本项目直接建立在 MAAP 之上，但研究的问题不同。

MAAP 证明 distributed wrist-camera observation 对 active perception 有帮助。它比较了 fixed global perception、single active perception 和 distributed multi-view wrist perception，并展示 multi-wrist perception 在 occlusion-heavy tasks 中尤其有效。

本项目进一步追问：什么样的表示能让模型理解并泛化这种行为。我们不只是把所有 wrist-camera image 输入 behavior-cloned controller，而是引入显式的 temporal mode supervision：

```text
哪条机械臂在观察？
哪条机械臂在操作？
这些角色什么时候应该切换？
模型能否在新场景中推断并使用这种结构？
```

在这个 framing 下，MAAP 提供环境、任务、专家 demonstration 和经验动机。本项目的贡献是在 MAAP 之上加入 mode-aware learning layer。

## 预期贡献

预期贡献包括：

1. **面向多智能体 active perception 的 per-arm action/perception label formulation。**
2. **带语义 mode label 的仿真和真机数据集。** 每条机械臂、每一帧都有 label。
3. **Learned mode predictor。** 从视觉和 state 中推断每条机器人何时应该观察、何时应该操作。
4. **Mode-conditioned VLA framework。** 测试显式 role/phase label 是否能提升 active-perception control。
5. **Mode-aware WAM framework。** 测试 label 是否能提升 future prediction 和 mode-sequence planning。
6. **系统性评估。** 分析显式 label 在什么情况下能超越 MAAP-style direct behavior cloning。

## 风险与应对

**风险 1：Label 有噪声或存在歧义。**

有些动作可能同时服务于 perception 和 manipulation。应对方式是，初始阶段使用严格定义：只有 contact/world-changing segment 标为 action；viewpoint setup 和 waiting 标为 perception。歧义帧可以忽略、降低权重，或者使用 soft label。

**风险 2：ModeNet label 预测很好，但 VLA 没有提升。**

这说明 mode prediction 本身不是主要瓶颈。项目需要测试多种 mode 注入方式：conditioning token、auxiliary loss、policy gating 和 hierarchical mode-first planning。

**风险 3：Ground-truth label 有帮助，但 predicted label 没有帮助。**

这说明 ModeNet error 是瓶颈。下一步可以引入 uncertainty-aware execution：当 ModeNet 不确定时，策略选择更安全的 perception action 或主动获取额外视角。

**风险 4：仿真 label 不能迁移到真机。**

这个 label space 是语义化且低维的，理论上应该比 raw action 更容易迁移。真机 pseudo-labeling 可以通过 gripper/contact signal、object tracking 和少量 human-labeled validation set 来改进。

## 项目里程碑

### Milestone 1：Labeled MAAP Dataset

- 为初始 MAAP tasks 生成 labeled trajectories。
- 用视频和 label viewer 验证 per-arm labels。
- 将 labels 导出到 VLA/WAM pipeline 使用的训练格式。

### Milestone 2：ModeNet

- 训练 vision-only 和 vision+state ModeNet。
- 评估 frame accuracy、F1 和 transition timing。
- 测试跨 seed、object pose 和 role assignment 的泛化。

### Milestone 3：Mode-Conditioned VLA

- 训练 no-label、ground-truth-label、predicted-label 和 auxiliary-label policies。
- 与 MAAP-style multi-wrist BC baselines 比较。
- 分析 mode label 是否减少 role-switching failure。

### Milestone 4：Mode-Aware WAM

- 训练带 per-arm mode label 和不带 mode label 的 WAM variants。
- 评估 future prediction 和 mode-sequence planning。
- 测试 WAM 是否能预测哪种 role allocation 更有利于任务进展。

### Milestone 5：Real-Robot Extension

- 采集带 weak labels 的真实轨迹。
- 在真实数据上 fine-tune 或 validate ModeNet。
- 评估 mode-aware VLA/WAM 是否提升真实 active perception。

## 总结

本项目的核心想法很简单：active perception 不应该只是 demonstration 里隐藏的行为。机器人应该学习自己什么时候是在看，什么时候是在动手操作。

MAAP 证明了 distributed wrist camera 可以在多臂操作中提供强 active perception。本项目在此基础上加入显式的 per-arm action/perception label，学习从视觉和 state 中预测这些 label，并测试这些 label 是否能帮助 VLA 和 WAM 获得更可解释、更可诊断、更可泛化的 active perception 能力。
