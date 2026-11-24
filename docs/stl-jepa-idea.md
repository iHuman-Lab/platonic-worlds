# STL-JEPA: Constraint-Space World Models

## Central hypothesis

> A predictive world model learned in constraint space can capture modality-independent
> physical and temporal relationships, allowing different observation modalities to access
> and predict the same underlying world dynamics.

## How the three ideas fit

| Idea | Role in this work |
| --- | --- |
| JEPA | Predictive mechanism: predict future latent constraint representations, not pixels |
| STL (Signal Temporal Logic) | Defines which physical and temporal properties the representation must preserve |
| Platonic Representation Hypothesis | Motivation for convergence across modalities |

Caveat: the Platonic hypothesis says different models may *independently* converge. We
*deliberately induce* convergence to a space defined by physical constraints and temporal
semantics. Call it **constraint-induced representational convergence**. We do not claim to
reproduce the Platonic hypothesis.

## Formulation (starting point)

Observations `x_t^(m)` from modality `m`:

- Encoder: `z_t^(m) = E_m(x_t^(m))`
- Constraint-space map: `c_t^(m) = G_m(z_t^(m))`
- Cross-modal alignment: `L_cross = sum_{m,n} || c_t^(m) - c_t^(n) ||^2`
- Physical constraints: `L_physical = sum_{m,j} || g_hat_j(c_t^(m)) - g_j(s_t) ||^2`
- STL semantics: `L_STL = sum_{m,phi} || rho_phi(C_{0:T}^(m)) - rho_phi(s_{0:T}) ||^2`
- JEPA prediction: `C_hat_{t+k} = F(C_t, u_{t:t+k-1})`, `L_pred = || C_hat_{t+k} - sg(C_{t+k}) ||^2`
- Total: `L = l1*L_cross + l2*L_physical + l3*L_STL + l4*L_pred`

## Proposed architecture

- **Teacher:** privileged robot state `s_t` -> constraint evaluator `G(s_t)` -> target `c_t*`
  (EMA teacher as in JEPA).
- **Student:** image `I_t` -> encoder `E(I_t)` -> `c_t`.
- **Predictor:** `c_hat_{t+k} = F(c_t, u_{t:t+k-1})`, predicting future constraint
  representations, not future pixels.
- **STL supervision:** `rho_phi(c_{0:T}) ~ rho_phi(s_{0:T})`.

## Critique points and responses

1. **Loss redundancy.** Matching constraint values does not imply matching latents
   (identifiability), so the losses play different roles. Alignment aligns the
   representations themselves, physical loss preserves constraint meaning, and STL preserves
   temporal task semantics. Treat the extra losses as ablations, not three independent
   contributions.
2. **Platonic claim.** Do not claim reproduction (see caveat above).
3. **JEPA is not just a name.** The model predicts future constraint representations, so the
   latent-prediction objective remains meaningful even with supervised anchors.
4. **Constraint set is the key design decision.** Start small.
5. **Experiments** (see below).

## Constraint categories

| Category | Examples |
| --- | --- |
| Geometric | Obstacle distance, workspace boundaries, relative position |
| Physical | Contact, grasp stability, velocity limits, collision |
| Task-related | Goal proximity, task progress, sequencing |

Example STL specs:

- Navigation: `phi = G[0,T](d_obstacle > d_min) AND F[0,T](d_goal < eps)`
- Manipulation: `phi = F[0,T](contact AND G[0,tau](grasp_stable))`

Implementation note: `rho_phi` must be differentiable (e.g. soft-min/soft-max over time) for
`L_STL` to be backpropagated.

## Experiments

| Experiment | Question |
| --- | --- |
| Constraint supervision ablation | Does STL add information beyond atomic physical-constraint regression? |
| Cross-modal transfer | Can a new modality use the trained constraint-space dynamics without retraining the world model? |
| Predictive evaluation | Does JEPA predict future constraint evolution better than conventional latent dynamics? |
| STL-aware planning | Does planning in constraint space improve task satisfaction and safety? |
| Frozen-encoder RL comparison | Does a small policy head learn faster and more safely on constraint-space features than on other representations? |

**Frozen-encoder RL comparison (downstream evaluation, not the training method):**

- Freeze the encoder and train a small policy head (PPO or SAC, linear or one-layer MLP) on `c_t`.
- Train the same head on: a random encoder, plain ConvJEPA, a raw-pixel CNN, and privileged
  state (upper bound).
- Reward: CarRacing tile reward, plus a variant using STL robustness `rho_phi`. Report return,
  return against environment steps, and constraint violations (off-road time, spins).
- Use at least 5 policy seeds per representation and several track seeds. Return is noisy.
- Keep the head small, so the encoder is what is being tested.
- Also run MPC with the learned `F` as the planner, to separate good features from good
  dynamics.
- Limits: it does not test cross-modal transfer (the frozen-`F` experiment does), and the tile
  reward does not penalize unsafe-but-fast driving.

**Key experiment:** train `F: (c_t, u_t) -> c_{t+1}` on robot-state encodings. Then attach a
new visual encoder `I_t -> c_t^I` and test whether `F(c_t^I, u_t)` predicts future constraint
states with `F` frozen. Co-training `F` with the image encoder, or fine-tuning it afterward,
invalidates the test.

## Versions

- **A: Constraint-supervised JEPA.** Robot state provides constraint labels. The visual
  encoder reproduces them and JEPA predicts their evolution. Easier to implement and
  validate. **Start here.**
- **B: Constraint-emergent JEPA.** Only trajectory-level STL supervision; the shared
  constraint representation is discovered. More ambitious and more novel.

Design A so that B is a natural extension.

## Contributions

1. Constraint-space representation of the physical world via task-relevant constraints.
2. STL-grounded predictive learning of constraint evolution.
3. Transferable world modeling: different observation pathways share one predictive model
   without retraining its dynamics.

## Candidate titles

- Learning a Common Constraint Space: Bridging Predictive World Models, Formal
  Specifications, and Multimodal Representations
- Constraint-Space World Models for Multimodal Embodied Intelligence

## Open question

Does the shared constraint space give a genuinely transferable predictive model, or is it just
a convenient way to regress physical state variables? The transfer experiment decides this.

## Planned implementation steps (Version A)

1. `constraints` section in `configs/config.yaml` (small racing-env set: road-edge distance,
   speed limit, progress).
2. `src/stl/constraints.py`: compute `g(s_t)` from saved state.
3. Differentiable STL robustness (soft-min/max).
4. EMA teacher producing `c*`.
5. Student visual encoder trained with alignment plus JEPA prediction in constraint space.
6. Transfer evaluation with frozen `F`.
