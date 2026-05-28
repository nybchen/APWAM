# Role-Switching Active Perception for Generalizable VLA/WAM

## Abstract

Existing active-perception manipulation systems usually learn from demonstrations with direct behavior cloning: given images, proprioception, and a task instruction, the model predicts robot actions. This can work in-distribution, but the learned policy does not explicitly know whether a robot arm is moving to gather information or moving to change the world. The distinction is especially important in multi-arm manipulation, where one arm may manipulate an object while another arm repositions its wrist camera, waits, or provides a task-relevant view.

This project studies whether explicit per-arm action/perception mode labels can turn active perception into a role-switching problem that is easier to learn and generalize. We collect trajectories from simulation, starting with RoboFactory/MAAP, and later from real robots. Each trajectory is annotated at every frame and for every arm with a binary label:

```text
z_t^i in {0, 1}

i: robot arm index
t: frame/action timestep
0: perception / observation / information gathering
1: action / manipulation / world-changing interaction
```

We then convert these per-frame labels into temporally coherent role stages, generate expert and corrupted candidate role/action futures, and train a scored WAM to assign high progress and advantage to expert futures and lower scores to corrupted alternatives. Finally, we use the high-score role configuration to condition Vision-Language-Action models (VLA) and World Action Models (WAM), and evaluate whether this improves transfer across tasks, layouts, occlusion patterns, robot roles, and real-world settings.

The central research question is:

```text
Can explicit per-arm action/perception labels turn active perception from an implicit behavior-cloning artifact into a learnable role-switching decision problem?
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
role-stage progress, and expert-versus-corrupted candidate ranking, they will learn
more generalizable active-perception behavior than models trained only with direct
behavior cloning.
```

The intuition is that perception and manipulation have different causal effects:

- In perception mode, an arm may move its wrist camera, reduce occlusion, reveal a target, or maintain a useful viewpoint. The image changes, but the task object usually should not move significantly.
- In action mode, an arm makes contact, grasps, pulls, pushes, lifts, places, or releases. The object state, articulation state, contact state, or task progress may change.

Direct BC mixes these regimes into one action distribution. A role-switching model can instead learn the structure of the task: who should look, who should act, when roles should switch, which candidate future makes progress, and how those switches affect future observations and states.

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

This project has five objectives.

1. **Build a labeled active-perception dataset.** Generate per-arm, per-frame action/perception labels from simulation and real-robot trajectories.
2. **Convert labels into role stages.** Detect switch events and compute role-stage progress from ground-truth demonstrations.
3. **Generate scored candidates.** Treat expert trajectories as high-score candidates and corrupted trajectories as lower-score alternatives.
4. **Train a scored candidate WAM.** Learn future prediction and candidate ranking jointly.
5. **Use high-score roles to improve VLA/WAM.** Test whether role-conditioned execution improves beyond direct behavior cloning and frame-wise mode prediction.

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

### 1. Mode Labels and Role Stages

The raw supervision is the per-arm action/perception mode sequence:

```text
Z_t = [z_t^0, z_t^1, ..., z_t^{N-1}]
```

The sequence is then compressed into temporally coherent role stages. A new stage starts whenever the role configuration changes:

```text
c_t = 1[Z_t != Z_{t-1}]
```

For example:

```text
t = 0..20    Z = [0,0]  -> role-stage 0
t = 21..55   Z = [1,0]  -> role-stage 1
t = 56..80   Z = [0,1]  -> role-stage 2
```

This changes the problem from frame-wise mode classification to role-switching: given the current observation history and current role, should the system stay in the same role configuration or switch to a new one?

### 2. Role Progress and Advantage

Expert trajectories with ground-truth action/perception labels define the ideal role-stage progress. If a trajectory contains `K` role stages, completing each stage contributes `1/K` progress:

```text
role-stage 0 completed -> +0.33
role-stage 1 completed -> +0.33
role-stage 2 completed -> +0.33
total                  -> 1.00
```

The role progress is:

```text
P_role(t) in [0, 1]
A_role(t, t') = P_role(t') - P_role(t)
```

#### Computing Timestamp Advantage from the Current H5 Trajectory Format

The current RoboFactory/MAAP trajectory format stores per-agent actions and mode labels under each episode group:

```text
traj_i/
  actions/
    panda_wristcam-0        # [T, action_dim]
    panda_wristcam-1        # [T, action_dim]
    panda_wristcam-2        # optional for three-arm tasks
  mode_labels/
    panda_wristcam-0        # [T], 0 = perception, 1 = action
    panda_wristcam-1        # [T]
    panda_wristcam-2        # optional
```

For each trajectory, first stack the per-agent labels into a role matrix:

```text
agent_keys = sorted(traj["actions"].keys())
Z_star[t, i] = traj["mode_labels"][agent_keys[i]][t]
Z_star shape = [T, N]
```

Then run-length encode the ground-truth role sequence into stages:

```text
stage 0: [s_0, e_0), role Z_0
stage 1: [s_1, e_1), role Z_1
...
stage K-1: [s_{K-1}, e_{K-1}), role Z_{K-1}

Z_t belongs to stage k if s_k <= t < e_k
```

The dense ground-truth role progress at every timestamp is:

```text
P_star[t] = (k + p_k(t)) / K

k        = ground-truth stage index at timestamp t
p_k(t)  = (t - s_k) / max(e_k - s_k - 1, 1)
K        = number of role stages
```

This gives a monotonic progress curve from `0` to `1`. The per-timestamp one-step advantage is:

```text
A_step_star[t] = P_star[min(t + 1, T - 1)] - P_star[t]
```

This dense progress gives small positive reward while moving inside a role stage. The main active-perception signal should be concentrated at successful stage transitions, so the score also includes an explicit switch or stage-completion bonus:

```text
B_switch_star[t] =
    1 / K, if t is a GT switch boundary and the new role is reached correctly
    0,     otherwise

A_frame_star[t] =
    alpha_dense * A_step_star[t]
    + alpha_switch * B_switch_star[t]
```

Use `alpha_switch > alpha_dense`. The intended behavior is that ordinary frames inside a stage receive small shaping advantage, while successfully switching from one role stage to the next receives a visibly larger advantage.

For a horizon `h`, the expert horizon advantage is:

```text
A_h_star[t, h] = P_star[min(t + h, T - 1)] - P_star[t]
```

For the ground-truth trajectory, the role correctness term is `1`, so its final training score is the full progress gain:

```text
score_star[t, h] =
    A_h_star[t, h]
    + alpha_switch * number_of_GT_switches_crossed(t, t+h) / K
```

In other words, the expert labeled trajectory is the high-score candidate at every timestamp. It receives full credit for progressing through the correct role stages with the correct switch timing and correct active arm assignment.

#### Re-Scoring Corrupted Candidates Against the Ground Truth

Corrupted candidates are generated by modifying the ground-truth role labels or short-horizon actions, but their scores are always computed against the original ground-truth trajectory. This is important: the corruption does not define a new target timeline. The original expert trajectory remains the reference.

For a corrupted role candidate `Z_tilde[t:t+h]`, compute a role-match quality:

```text
q_role[t, h] =
    mean_{u=t}^{t+h} (
        1 - HammingDistance(Z_tilde[u], Z_star[u]) / N
    )
```

Then compute a switch quality term. Let `B_star` be the set of ground-truth switch times where `Z_star[t] != Z_star[t-1]`, and let `B_tilde` be the candidate switch times. A simple version is:

```text
q_switch[t, h] = 1.0
```

unless the horizon contains a switch. If the ground truth switches inside the horizon, the candidate is rewarded for switching to the correct next role near the correct time:

```text
q_switch =
    exp(-abs(tau_tilde - tau_star) / sigma)
    * (1 - HammingDistance(Z_tilde[tau_tilde], Z_star[tau_star]) / N)
```

If the ground truth stays but the candidate switches, `q_switch` is penalized. If the ground truth switches but the candidate stays, `q_switch` is also penalized:

```text
ground truth stays, candidate switches -> q_switch = gamma_spurious
ground truth switches, candidate stays -> q_switch = gamma_missed
```

where `gamma_spurious` and `gamma_missed` are small values such as `0.0` to `0.3`.

If the candidate also corrupts actions, add an action quality term for the active manipulation arm:

```text
q_action[t, h] =
    exp(-mean_action_error(active arms, t:t+h) / sigma_a)
```

For pure role corruption experiments, set `q_action = 1`.

The corrupted candidate advantage is then:

```text
score_tilde[t, h] =
    A_h_star[t, h]
    * q_role[t, h]
    * q_switch[t, h]
    * q_action[t, h]
    + alpha_switch * B_switch_tilde[t, h]
```

where `B_switch_tilde[t, h]` is nonzero only when the candidate crosses the same GT switch boundary, switches to the correct next role, and does so within the timing tolerance:

```text
B_switch_tilde[t, h] =
    (1 / K)
    * exp(-abs(tau_tilde - tau_star) / sigma)
    * (1 - HammingDistance(Z_tilde[tau_tilde], Z_star[tau_star]) / N)
```

This means a corrupted sequence can still receive partial credit if it is close to the expert, but successful stage switching receives much larger advantage than ordinary within-stage frames. For example, a candidate that switches to the correct role two frames late should score lower than the exact expert switch but higher than one that assigns manipulation to the wrong arm. This is the main reason to use a soft advantage rather than a binary expert/non-expert label.

#### Candidate Corruption Operators

The corruption process should include both easy negatives and hard negatives:

```text
Input:
  Z_star: [T, N] ground-truth role labels
  A_star: dict of per-agent actions
  stages: run-length encoded GT role stages

Output:
  Z_tilde: corrupted role labels
  A_tilde: optional corrupted actions
  score_tilde[t, h]: GT-referenced advantage score
```

Useful role corruptions:

- **Random role frame flip:** choose a segment and flip one arm's mode.
- **Wrong active arm:** in an action stage, move the `1` label from the GT active arm to another arm.
- **Early switch:** shift a GT switch boundary earlier by `delta` frames.
- **Delayed switch:** shift a GT switch boundary later by `delta` frames.
- **Missed switch:** keep the previous role across a GT switch boundary.
- **Spurious switch:** insert a role switch where the GT role stays constant.
- **Stage permutation:** swap two role stages while keeping the same stage lengths.
- **All-perception collapse:** set a manipulation stage to all zeros.
- **All-action collapse:** set multiple arms to action when only one arm should manipulate.

Useful action corruptions:

- **Wrong-arm action:** keep the correct role label but use another arm's action segment.
- **Temporal action shift:** use the expert action from `t + delta` instead of `t`.
- **No-op active arm:** replace the active arm's action with hold-position actions.
- **Noisy action:** add bounded noise to the active arm's action segment.

The recommended first implementation is:

```text
1. Read Z_star from traj_i/mode_labels/<agent_key>.
2. Extract role stages by run-length encoding Z_star.
3. Compute P_star[t], A_step_star[t], and A_h_star[t,h].
4. Generate M corrupted candidates per expert trajectory.
5. For each candidate, compute q_role, q_switch, optional q_action.
6. Store or train with (history, candidate_role, candidate_action, score_tilde).
7. Train WAM with progress regression and expert-vs-corruption ranking loss.
```

The ranking loss should compare the expert candidate against corrupted candidates from the same timestamp and horizon:

```text
L_rank =
    -log sigmoid(
        score_theta(history_t, Z_star[t:t+h], A_star[t:t+h])
        - score_theta(history_t, Z_tilde[t:t+h], A_tilde[t:t+h])
    )
```

This gives the WAM the exact behavior we want at inference: given several candidate role/action futures, assign the highest score to the one that best follows the ground-truth role-stage progress structure. The model is not only trained to predict future observations or states; it is also trained to assign higher value to role/action candidates that make task progress.

### 3. Scored Candidate WAM

A mode-aware WAM receives observation history, robot state, language, the previous role configuration, and a candidate future role/action segment:

```text
WAM_theta(O_{t-H:t}, q_{t-H:t}, language, Z_{t-1}, Z', a_{t:t+h})
    -> future observations/states
    -> P_role(t+h)
    -> A_role(t, t+h)
```

The ground-truth trajectory provides the full-score candidate. Corrupted trajectories provide lower-score alternatives. The training objective can combine world prediction and candidate scoring:

```text
L = L_world + lambda_progress * L_progress + lambda_rank * L_rank
```

where `L_world` predicts future observations, robot states, object states, or task progress; `L_progress` regresses the role progress or advantage score; and `L_rank` forces expert candidates to score above corrupted candidates:

```text
L_rank =
    -log sigmoid(
        score_theta(expert candidate)
        - score_theta(corrupted candidate)
    )
```

This formulation makes the WAM an evaluator of candidate role/action futures, rather than only a passive next-frame predictor.

### 4. Inference with High-Score Candidates

At inference time, the system enumerates or samples candidate next role configurations and short-horizon action candidates:

```text
for Z' in {0,1}^N:
    score(Z') = WAM_theta(history, Z_{t-1}, Z')

if max score is high enough:
    switch to argmax_Z' score(Z')
else:
    stay in current role
```

The selected high-score role configuration is then passed to the low-level VLA or policy:

```text
pi(a_t | O_t, q_t, language, Z_t)
```

The important distinction is that the model does not need a separate generic "active model." Active perception is represented as choosing a role configuration that improves expected role-stage progress under occlusion and multi-arm coordination constraints.

### 5. ModeNet as a Baseline or Auxiliary Module

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

ModeNet should be evaluated not only by frame accuracy, but also by transition quality. The hardest and most important moments are the boundaries: when an arm stops observing and starts manipulating, or when manipulation responsibility transfers from one arm to another. In the updated framework, ModeNet is a useful baseline or auxiliary head, but the main contribution is the scored role-switching WAM.

### 6. Role-Conditioned VLA

A role-conditioned VLA predicts actions with an explicit role or mode input:

```text
pi(a_t^i | obs_t, q_t, language, agent_id=i, z_t^i)
```

During training, the policy can use ground-truth or pseudo-label modes. During inference, it can use either ModeNet predictions as a baseline or WAM-selected high-score roles as the main method:

```text
Z_hat_t or Z_WAM_t -> VLA -> a_t
```

Several designs should be compared:

- **No-label VLA:** standard behavior cloning baseline.
- **Ground-truth mode VLA:** upper bound using true labels at test time in simulation.
- **Predicted-mode VLA:** baseline using ModeNet predictions.
- **WAM-selected-role VLA:** practical model using the scored candidate WAM to choose the role configuration.
- **Auxiliary-head VLA:** one network predicts both action and mode:

```text
L = L_action + lambda * L_mode
```

Role labels can be injected as binary tokens, per-arm embeddings, adapter conditions, or high-level gating variables. The main question is not only whether success improves, but whether failures become more interpretable: wrong role selection should correspond to understandable errors such as observing too long, manipulating too early, or assigning the wrong arm to the manipulation phase.

### 7. Mode-Aware Dynamics

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

The scored candidate WAM extends this dynamics model with a progress/advantage head, allowing it to rank candidate role sequences before the VLA generates or executes detailed low-level actions.

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

This experiment is a baseline and diagnostic check. High ModeNet accuracy alone is not the final goal; the more important question is whether the system can make temporally coherent role-switching decisions.

### Experiment 3: Can the WAM Rank Expert Role Futures Above Corrupted Futures?

Train the scored candidate WAM on ground-truth role/action segments and corrupted alternatives.

Positive candidates:

- expert role sequence with ground-truth action/perception labels
- expert switch timing
- expert arm assignment for manipulation stages
- expert short-horizon action segment

Negative or lower-score candidates:

- wrong role configuration
- wrong active arm
- premature switch
- delayed switch
- missed switch
- unnecessary switch
- corrupted action segment under the correct role

Metrics:

- progress/advantage regression error
- expert-vs-corruption ranking accuracy
- AUC for choosing expert candidate over corrupted candidate
- switch timing error
- wrong-arm rejection rate
- missed-switch detection rate

Expected outcome:

```text
The expert trajectory should receive the highest role progress and advantage.
Corrupted role/action futures should receive lower scores according to how much
they damage task progress.
```

### Experiment 4: Do Role Labels and WAM Scoring Improve VLA Active Perception?

Train VLA/BC policies with and without mode labels.

Baselines:

- global fixed view BC
- single active wrist view BC
- MAAP-style multi-wrist BC without labels
- role-conditioned VLA with ground-truth labels
- role-conditioned VLA with predicted labels
- auxiliary-mode VLA
- role-conditioned VLA with WAM-selected high-score roles

Metrics:

- task success rate
- success under new object poses/layouts
- success under stronger occlusion
- number and type of role-switching failures
- action smoothness and unnecessary camera motion
- recovery after partial occlusion or failed first attempt

Expected outcome:

```text
Role-conditioned policies should improve most on occlusion-heavy and multi-phase tasks,
where the timing of observe/action role switches matters. WAM-selected high-score
roles should reduce premature switches, delayed switches, and wrong-arm assignments.
```

### Experiment 5: Do Labels Improve WAM Prediction and Planning?

Train WAM variants with and without mode conditioning.

Baselines:

- WAM without mode labels
- WAM with a single global mode label
- WAM with per-arm mode labels
- WAM with predicted per-arm mode labels
- scored candidate WAM with expert and corrupted role/action candidates

Metrics:

- future image prediction error
- object pose/articulation prediction error
- contact/task-progress prediction accuracy
- quality of predicted information gain
- success rate when planning over mode sequences
- candidate ranking accuracy
- selected-role success rate

Expected outcome:

```text
Per-arm mode labels should help WAM distinguish viewpoint-changing actions from
world-changing actions. Adding progress and ranking supervision should make the
WAM useful for choosing role futures, not only predicting them.
```

### Experiment 6: Sim-to-Real and Real-Only Evaluation

Transfer the learned role scorer and role-conditioned policy to real robot data.

Evaluation settings:

- train role-stage scoring in simulation, test on real demonstrations
- fine-tune the scored candidate WAM with a small amount of real pseudo-labeled data
- train VLA/WAM with simulated labels and evaluate real active-perception behavior
- compare pseudo-labels against small human-labeled validation clips

Metrics:

- real-data role-stage and switch prediction accuracy
- expert-vs-corruption ranking accuracy on real clips
- active-perception success under occlusion
- robustness to camera calibration differences
- sim-to-real label transfer quality

## Relationship to MAAP

This project builds directly on MAAP but asks a different question.

MAAP shows that distributed wrist-camera observations are useful for active perception. It compares fixed global perception, single active perception, and distributed multi-view wrist perception, and demonstrates that multi-wrist perception is especially helpful for occlusion-heavy tasks.

This project asks what representation should allow a model to understand and generalize that behavior. Instead of only feeding all wrist-camera images into a behavior-cloned controller, we introduce explicit temporal role supervision and a WAM that scores candidate role futures:

```text
Which arm is observing?
Which arm is manipulating?
When should those roles switch?
Which candidate future role/action segment makes the most task progress?
Can a model infer, rank, and use this structure in new situations?
```

In this framing, MAAP provides the environment, tasks, expert demonstrations, and empirical motivation. The proposed contribution is the role-switching and scored-candidate WAM layer on top of MAAP.

## Expected Contributions

The expected contributions are:

1. **A per-arm action/perception label formulation** for multi-agent active perception.
2. **A labeled simulation and real-robot dataset** where each arm and each frame has a semantic mode label.
3. **A role-stage progress and advantage formulation** that gives expert role/action futures high score and corrupted futures lower score.
4. **A scored candidate WAM** that predicts future dynamics while ranking candidate role/action segments by expected task progress.
5. **A role-conditioned VLA framework** that executes the high-score role configuration selected by the WAM.
6. **A systematic evaluation** showing when explicit role switching helps beyond MAAP-style direct behavior cloning and frame-wise mode prediction.

## Risks and Mitigations

**Risk 1: Labels are noisy or ambiguous.**

Some motions may serve both perception and manipulation. To mitigate this, labels should initially use a strict definition: only contact/world-changing segments are action; viewpoint setup and waiting are perception. Ambiguous frames can be ignored, down-weighted, or modeled with soft labels.

**Risk 2: The WAM learns dynamics but not useful candidate ranking.**

This would mean next-state prediction alone is too weak as supervision. The mitigation is to include explicit progress regression and ranking loss between expert and corrupted candidates, instead of training the WAM only as a passive predictor.

**Risk 3: The WAM selects high-score roles, but the low-level policy cannot execute them.**

This would identify the VLA/control layer as the bottleneck. The project should compare ground-truth-role VLA, WAM-selected-role VLA, and oracle-action rollouts to separate role selection errors from execution errors.

**Risk 4: Simulation labels do not transfer to real robots.**

The label space is intentionally semantic and low-dimensional, which should transfer better than raw actions. Real-world pseudo-labeling can be improved with gripper/contact signals, object tracking, and a small human-labeled validation set.

**Risk 5: Corrupted candidates are too easy or unrealistic.**

If negative candidates are trivial, the WAM may learn shortcuts that do not help at deployment. Corruptions should include hard negatives such as slightly early switches, slightly delayed switches, wrong-arm assignments that are visually plausible, and correct roles paired with bad short-horizon actions.

## Project Milestones

### Milestone 1: Labeled MAAP Dataset

- Generate labeled trajectories for the initial MAAP tasks.
- Validate per-arm labels with videos and the label viewer.
- Export labels into the training format used by VLA/WAM pipelines.

### Milestone 2: Role Stages and Candidate Generation

- Convert per-arm label sequences into role stages and switch events.
- Generate expert candidates and hard corrupted candidates.
- Assign role progress and advantage scores to candidate segments.

### Milestone 3: ModeNet Baseline

- Train vision-only and vision+state ModeNet.
- Evaluate frame accuracy, F1, and transition timing.
- Test generalization across seeds, object poses, and role assignments.

### Milestone 4: Scored Candidate WAM

- Train WAM variants with world prediction, progress regression, and ranking loss.
- Evaluate whether expert candidates rank above corrupted candidates.
- Test candidate role selection under held-out poses, occlusions, and role assignments.

### Milestone 5: Role-Conditioned VLA

- Train no-label, ground-truth-role, predicted-role, and WAM-selected-role policies.
- Compare against MAAP-style multi-wrist BC baselines.
- Analyze whether role switching reduces premature, delayed, missed, and wrong-arm failures.

### Milestone 6: Real-Robot Extension

- Collect real trajectories with weak labels.
- Fine-tune or validate role-stage prediction and WAM candidate scoring on real data.
- Evaluate whether WAM-selected role-conditioned VLA improves real active perception.

## Summary

The core idea of this project is simple: active perception should not only be a behavior hidden inside demonstrations. A robot should learn which arm is looking, which arm is acting, and when the team should switch roles.

MAAP demonstrates that distributed wrist cameras can provide strong active perception in multi-arm manipulation. This project extends that result by adding explicit per-arm action/perception labels, converting them into role stages, training a WAM to rank expert role/action futures above corrupted alternatives, and using high-score role candidates to condition VLA/WAM execution. In this framing, active perception is not a separate active model; it is a role-switching decision problem grounded in world-model progress.
