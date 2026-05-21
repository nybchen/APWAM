# Mode-Aware Active Perception for Generalizable VLA/WAM

## Abstract

Existing active-perception manipulation systems usually learn from demonstrations with direct behavior cloning: given images, proprioception, and a task instruction, the model predicts robot actions. This can work in-distribution, but the learned policy does not explicitly know whether a robot arm is moving to gather information or moving to change the world. The distinction is especially important in multi-arm manipulation, where one arm may manipulate an object while another arm repositions its wrist camera, waits, or provides a task-relevant view.

This project studies whether explicit per-arm action/perception mode labels can make active perception more generalizable. We collect trajectories from simulation, starting with RoboFactory/MAAP, and later from real robots. Each trajectory is annotated at every frame and for every arm with a binary label:

```text
z_t^i in {0, 1}

i: robot arm index
t: frame/action timestep
0: perception / observation / information gathering
1: action / manipulation / world-changing interaction
```

We then train a model to infer these labels from visual observations, or from vision plus robot state. Finally, we evaluate whether these labels help Vision-Language-Action models (VLA) and World Action Models (WAM) learn active perception that transfers across tasks, layouts, occlusion patterns, robot roles, and real-world settings.

The central research question is:

```text
Can explicit per-arm action/perception labels turn active perception from an implicit behavior-cloning artifact into a learnable, interpretable, and generalizable capability?
```

## Background and Motivation

MAAP proposes a useful starting point for this project. The MAAP IROS paper argues that multi-agent manipulation naturally creates distributed active perception: every manipulation arm has a wrist camera, and these cameras can provide task-relevant viewpoints while the arms execute the task. In the MAAP experiments, distributed multi-wrist perception improves over fixed global perception and often over a single active wrist view, especially in severe occlusion and multi-phase tasks such as opening a microwave and retrieving an object.

However, MAAP still relies heavily on imitation learning from expert trajectories. The controller receives observations and directly predicts actions. Temporal role switching exists in the demonstrations: one arm may observe, another may open a door, then a different arm may retrieve the object. But this role structure is not explicitly represented in the learned policy. The model is expected to discover from behavior cloning when each arm should look, wait, move to a viewing pose, grasp, pull, lift, or place.

This creates three limitations:

1. **Implicit active perception.** The policy may reproduce observed motions, but it has no explicit variable that says "this arm is currently gathering information" versus "this arm is currently manipulating the environment."
2. **Poor diagnostics.** When the policy fails, it is difficult to tell whether it chose the wrong role, used the wrong view, or generated a bad low-level action.
3. **Weak generalization.** Direct behavior cloning can overfit to task-specific trajectories and may not transfer role-switching behavior to new layouts, occlusions, object poses, or robot assignments.

The proposed project addresses this gap by making temporal role switching explicit.

## Core Hypothesis

The hypothesis is:

```text
If VLA and WAM models are trained with explicit per-arm action/perception mode labels,
they will learn more generalizable active-perception behavior than models trained only
with direct behavior cloning.
```

The intuition is that perception and manipulation have different causal effects:

- In perception mode, an arm may move its wrist camera, reduce occlusion, reveal a target, or maintain a useful viewpoint. The image changes, but the task object usually should not move significantly.
- In action mode, an arm makes contact, grasps, pulls, pushes, lifts, places, or releases. The object state, articulation state, contact state, or task progress may change.

Direct BC mixes these regimes into one action distribution. A mode-aware model can instead learn the structure of the task: who should look, who should act, when roles should switch, and how those switches affect future observations and states.

## Label Definition

The label is per-arm and per-frame:

```text
Z_t = [z_t^0, z_t^1, ..., z_t^{N-1}]
```

For example, in a three-arm microwave task:

```text
[0, 0, 1] = arm 2 is manipulating; arms 0 and 1 are observing or waiting
[0, 1, 0] = arm 1 is manipulating; arms 0 and 2 are observing or waiting
```

`z_t^i = 1` means the arm is currently responsible for a world-changing manipulation segment:

- grasping or closing the gripper on an object or handle
- pulling, pushing, opening, closing, lifting, moving, placing, or releasing an object
- maintaining contact as part of task progress

`z_t^i = 0` means the arm is currently serving a perception, support, or non-manipulation role:

- moving to a viewing pose
- aiming a wrist camera at a task-relevant region
- waiting for another arm to finish a precondition
- holding a posture that provides visual context
- executing setup motion that does not contact or change the object

This definition is semantic, not purely kinematic. A robot can have nonzero joint actions while still being in perception mode if the motion is for viewpoint acquisition rather than object manipulation.

## Research Objectives

This project has four objectives.

1. **Build a labeled active-perception dataset.** Generate per-arm, per-frame action/perception labels from simulation and real-robot trajectories.
2. **Learn a mode predictor.** Train a model that predicts each arm's action/perception mode from visual observations, or from visual observations plus robot state.
3. **Use labels to improve VLA.** Test whether mode labels help policies learn when to observe and when to manipulate, beyond direct behavior cloning.
4. **Use labels to improve WAM.** Test whether mode labels help world/action models predict the consequences of perception actions versus manipulation actions.

## Data Collection and Label Generation

### Simulation Data

The first testbed is RoboFactory/MAAP because it already contains multi-arm manipulation tasks with wrist cameras, planner-generated demonstrations, and active-perception behavior. Initial tasks include:

- `PlaceCubeOnCart`
- `PickMeatFromPot`
- `TwoRobotsStackCubeActive`
- `PickMeatFromMicrowave`

These tasks cover simple visibility, localized occlusion, severe occlusion, multi-phase manipulation, and temporal role switching. The MAAP paper also provides a useful baseline suite: global fixed view, single active wrist view, and distributed multi-wrist perception.

Simulation labels can be generated from multiple signals:

- scripted planner phase
- active robot or `move_id`
- gripper open/close commands
- contact state
- object pose or articulation state change
- task progress events
- camera/viewpoint movement without contact

The first version can use planner-derived pseudo-labels, because the scripted expert already knows which arm is opening, retrieving, placing, or observing. Later versions can combine planner signals with contact and object-state changes to reduce label noise.

### Real-Robot Data

The second stage extends the same label definition to real robots. Real demonstrations may come from teleoperation, scripted collection, or human-guided active perception.

Possible label signals include:

- robot proprioception and end-effector motion
- gripper command and gripper width
- force/torque or tactile contact
- object tracking from external cameras or wrist cameras
- teleoperation logs
- human annotations for a small validation subset

The goal is not to manually label every frame. Instead, the project should use weak supervision and pseudo-labeling, then train a learned labeler that can transfer across tasks and platforms.

## Method

### 1. ModeNet: Learning When Each Arm Should Observe or Act

ModeNet predicts the per-arm mode vector:

```text
Z_hat_t = ModeNet(o_{t-k:t}, q_{t-k:t}, a_{t-k:t-1}, language)
```

where:

- `o_{t-k:t}` is a history of wrist/global images
- `q_{t-k:t}` is robot proprioception
- `a_{t-k:t-1}` is optional previous action history
- `language` is the task instruction or task id
- `Z_hat_t` contains one binary prediction per arm

Two input settings should be studied:

1. **Vision-only mode prediction.**
   This tests whether the label can be inferred from visual context alone.
2. **Vision + state mode prediction.**
   This tests whether proprioception, gripper state, and recent motion improve role/phase inference.

ModeNet should be evaluated not only by frame accuracy, but also by transition quality. The hardest and most important moments are the boundaries: when an arm stops observing and starts manipulating, or when manipulation responsibility transfers from one arm to another.

### 2. Mode-Conditioned VLA

A mode-conditioned VLA predicts actions with an explicit mode input:

```text
pi(a_t^i | obs_t, q_t, language, agent_id=i, z_t^i)
```

During training, the policy can use ground-truth or pseudo-label modes. During inference, it uses ModeNet predictions:

```text
z_hat_t^i -> VLA -> a_t^i
```

Several designs should be compared:

- **No-label VLA:** standard behavior cloning baseline.
- **Ground-truth mode VLA:** upper bound using true labels at test time in simulation.
- **Predicted-mode VLA:** practical model using ModeNet predictions.
- **Auxiliary-head VLA:** one network predicts both action and mode:

```text
L = L_action + lambda * L_mode
```

Mode labels can be injected as binary tokens, per-arm embeddings, adapter conditions, or high-level gating variables. The main question is not only whether success improves, but whether failures become more interpretable: wrong mode prediction should correspond to understandable errors such as observing too long, manipulating too early, or assigning the wrong arm to the manipulation phase.

### 3. Mode-Aware WAM

A WAM predicts future observations and states from current observations, actions, and mode labels:

```text
WAM(o_t, q_t, a_t, Z_t) -> o_{t+1:t+H}, q_{t+1:t+H}, s_{t+1:t+H}
```

The mode vector tells the world model how to interpret actions:

- If `z_t^i = 0`, the arm's motion should mainly affect camera pose, occlusion, and information quality.
- If `z_t^i = 1`, the arm's motion may affect object pose, contact, articulation, and task progress.

This can reduce dynamics ambiguity. The same low-level arm motion can mean very different things depending on whether it is approaching a viewing pose or executing an object interaction.

Mode labels also enable high-level planning over mode sequences:

```text
[0,0,0] -> [0,0,1] -> [0,1,0]
setup/viewing -> door opening -> object retrieval
```

The WAM can score whether a proposed role sequence is likely to reveal the right information and complete the task before the VLA generates the detailed low-level actions.

## Experimental Plan

### Experiment 1: Are the Labels Reliable?

Generate labeled trajectories in RoboFactory/MAAP and verify frame-level alignment with a label viewer.

Metrics:

- label consistency across trajectories
- agreement between planner labels and contact/object-change heuristics
- transition-frame correctness
- human spot-check accuracy on sampled videos

Expected outcome:

```text
Planner-derived pseudo-labels should be reliable enough to train an initial ModeNet,
especially for scripted simulation trajectories.
```

### Experiment 2: Can ModeNet Predict Action/Perception Modes?

Train ModeNet on labeled trajectories and evaluate generalization.

Splits:

- held-out random seeds
- held-out object poses
- held-out camera/object occlusion patterns
- held-out robot-role assignments
- eventually held-out tasks

Metrics:

- per-arm accuracy
- F1 score for action mode
- transition timing error
- false action rate during perception phases
- false perception rate during manipulation phases

Key ablations:

- vision only
- vision + proprioception
- vision + proprioception + action history
- single-view input
- multi-wrist input
- frozen visual encoder versus trainable visual encoder

### Experiment 3: Do Labels Improve VLA Active Perception?

Train VLA/BC policies with and without mode labels.

Baselines:

- global fixed view BC
- single active wrist view BC
- MAAP-style multi-wrist BC without labels
- mode-conditioned VLA with ground-truth labels
- mode-conditioned VLA with predicted labels
- auxiliary-mode VLA

Metrics:

- task success rate
- success under new object poses/layouts
- success under stronger occlusion
- number and type of role-switching failures
- action smoothness and unnecessary camera motion
- recovery after partial occlusion or failed first attempt

Expected outcome:

```text
Mode-conditioned policies should improve most on occlusion-heavy and multi-phase tasks,
where the timing of observe/action role switches matters.
```

### Experiment 4: Do Labels Improve WAM Prediction and Planning?

Train WAM variants with and without mode conditioning.

Baselines:

- WAM without mode labels
- WAM with a single global mode label
- WAM with per-arm mode labels
- WAM with predicted per-arm mode labels

Metrics:

- future image prediction error
- object pose/articulation prediction error
- contact/task-progress prediction accuracy
- quality of predicted information gain
- success rate when planning over mode sequences

Expected outcome:

```text
Per-arm mode labels should help WAM distinguish viewpoint-changing actions from
world-changing actions, improving both prediction and planning.
```

### Experiment 5: Sim-to-Real and Real-Only Evaluation

Transfer the learned labeler and mode-conditioned policy to real robot data.

Evaluation settings:

- train ModeNet in simulation, test on real demonstrations
- fine-tune ModeNet with a small amount of real pseudo-labeled data
- train VLA/WAM with simulated labels and evaluate real active-perception behavior
- compare pseudo-labels against small human-labeled validation clips

Metrics:

- real-data mode prediction accuracy
- active-perception success under occlusion
- robustness to camera calibration differences
- sim-to-real label transfer quality

## Relationship to MAAP

This project builds directly on MAAP but asks a different question.

MAAP shows that distributed wrist-camera observations are useful for active perception. It compares fixed global perception, single active perception, and distributed multi-view wrist perception, and demonstrates that multi-wrist perception is especially helpful for occlusion-heavy tasks.

This project asks what representation should allow a model to understand and generalize that behavior. Instead of only feeding all wrist-camera images into a behavior-cloned controller, we introduce explicit temporal mode supervision:

```text
Which arm is observing?
Which arm is manipulating?
When should those roles switch?
Can a model infer and use this structure in new situations?
```

In this framing, MAAP provides the environment, tasks, expert demonstrations, and empirical motivation. The proposed contribution is the mode-aware learning layer on top of MAAP.

## Expected Contributions

The expected contributions are:

1. **A per-arm action/perception label formulation** for multi-agent active perception.
2. **A labeled simulation and real-robot dataset** where each arm and each frame has a semantic mode label.
3. **A learned mode predictor** that infers when each robot should observe or act from vision and state.
4. **A mode-conditioned VLA framework** that tests whether explicit role/phase labels improve active-perception control.
5. **A mode-aware WAM framework** that tests whether labels improve future prediction and mode-sequence planning.
6. **A systematic evaluation** showing when explicit labels help beyond MAAP-style direct behavior cloning.

## Risks and Mitigations

**Risk 1: Labels are noisy or ambiguous.**

Some motions may serve both perception and manipulation. To mitigate this, labels should initially use a strict definition: only contact/world-changing segments are action; viewpoint setup and waiting are perception. Ambiguous frames can be ignored, down-weighted, or modeled with soft labels.

**Risk 2: ModeNet predicts labels well but VLA does not improve.**

This would mean mode prediction alone is not the bottleneck. The project should test multiple ways of injecting mode information: conditioning tokens, auxiliary loss, policy gating, and hierarchical mode-first planning.

**Risk 3: Ground-truth labels help but predicted labels do not.**

This would identify ModeNet errors as the bottleneck. The next step would be uncertainty-aware execution: when ModeNet is uncertain, the policy can choose safer perception actions or request additional views.

**Risk 4: Simulation labels do not transfer to real robots.**

The label space is intentionally semantic and low-dimensional, which should transfer better than raw actions. Real-world pseudo-labeling can be improved with gripper/contact signals, object tracking, and a small human-labeled validation set.

## Project Milestones

### Milestone 1: Labeled MAAP Dataset

- Generate labeled trajectories for the initial MAAP tasks.
- Validate per-arm labels with videos and the label viewer.
- Export labels into the training format used by VLA/WAM pipelines.

### Milestone 2: ModeNet

- Train vision-only and vision+state ModeNet.
- Evaluate frame accuracy, F1, and transition timing.
- Test generalization across seeds, object poses, and role assignments.

### Milestone 3: Mode-Conditioned VLA

- Train no-label, ground-truth-label, predicted-label, and auxiliary-label policies.
- Compare against MAAP-style multi-wrist BC baselines.
- Analyze whether mode labels reduce role-switching failures.

### Milestone 4: Mode-Aware WAM

- Train WAM variants with and without per-arm mode labels.
- Evaluate future prediction and mode-sequence planning.
- Test whether WAM can predict which role allocation improves task progress.

### Milestone 5: Real-Robot Extension

- Collect real trajectories with weak labels.
- Fine-tune or validate ModeNet on real data.
- Evaluate whether mode-aware VLA/WAM improves real active perception.

## Summary

The core idea of this project is simple: active perception should not only be a behavior hidden inside demonstrations. A robot should learn when it is looking and when it is acting.

MAAP demonstrates that distributed wrist cameras can provide strong active perception in multi-arm manipulation. This project extends that result by adding explicit per-arm action/perception labels, learning to predict those labels from vision and state, and testing whether they help VLA and WAM models acquire active perception that is more interpretable, diagnosable, and generalizable.
