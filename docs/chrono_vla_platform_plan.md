# Chrono-Based Active Suspension VLA Platform Plan

## 1. Core Answer

Project Chrono can provide visual and geometric terrain sensing, but it is not a ready-made SimplerEnv-style VLA benchmark for active suspension. Chrono is better treated as a future high-fidelity physics and sensor backend, not the current thesis control route.

Current decision: **do not make Chrono vision/VLA the main thesis direction**. The thesis should directly use already perceived road unevenness, i.e. the road preview height sequence or road roughness features, to design and evaluate active-suspension controllers.

For this thesis, the practical route is:

```text
road preview / road unevenness sequence
        -> BP-NMPS MPC gain scheduling
        -> SI-RPPO safe-teacher-gated force residuals
        -> active suspension control
```

## 2. Does Chrono Have Visual Road Information?

Yes. Chrono::Sensor supports RGB cameras, depth cameras, segmentation cameras, normal-map cameras, lidar, GPS, and IMU-like sensors. These sensors can be attached to Chrono bodies and are compatible with Chrono::Vehicle and SCMTerrain. This means a vehicle can carry forward-facing cameras or lidar to observe road geometry, terrain, and obstacles.

Chrono::Vehicle also supports rigid terrain, height-map terrain, and deformable terrain such as SCM terrain. Therefore, road information can be obtained in at least three forms:

- privileged terrain state: analytical height profile, height map, or terrain mesh used by the simulator;
- simulated perception: RGB/depth/segmentation/lidar from Chrono::Sensor;
- control abstraction: road preview sequence, road roughness class, local elevation map, or terrain embedding.

For active suspension, the third form is the most useful for control. Vision should be an optional upstream module, not the core claim unless enough data and perception evaluation are added.

## 3. Can We Build a SimplerEnv-Like Platform?

Yes, but it should be a domain-specific platform rather than a direct clone.

SimplerEnv is designed around manipulation policies: image observations, language instructions, robot arm actions, and task success. Active suspension has a different objective:

- control frequency is much higher;
- action is continuous force or controller parameter scheduling;
- success is not object manipulation but comfort, safety, tire load, actuator smoothness, and robustness;
- language is usually not required at every step; it is better used as a task/condition descriptor.

The active-suspension analogue should be:

```text
SimpleSuspensionEnv
  observation:
    - camera/depth/lidar road view, optional
    - road preview sequence or local elevation map
    - vehicle state history
    - actuator state
    - language or symbolic task tag
  action:
    - BP-NMPS gain residuals, preferred first
    - SI-RPPO safe-teacher-gated force residuals
    - direct four-corner force commands, only after safety wrappers
  reward/evaluation:
    - body/pitch/roll acceleration
    - suspension travel
    - tire load
    - unsafe steps
    - actuator tracking and action smoothness
    - deployment-gate non-regression
```

## 4. VLA Status For This Thesis

Do not start with a general OpenVLA-style 7B model. That is expensive, mismatched to the current evidence, and would shift the thesis from control algorithm design to perception-platform engineering. If a VLA-like interface is mentioned, keep it as future work:

```text
Input:
  road preview sequence
  vehicle state history
  optional task token, e.g. "comfort priority on rough road"

Output:
  7D MPC gain residual for BP-NMPS
  or safe gate/residual action for SI-RPPO
```

This keeps any future multimodal model as an adaptive controller interface instead of an unconstrained end-to-end force generator.

## 5. Staged Implementation Plan

### Stage A: Direct road-preview controller, current thesis priority

- Use existing road preview and vehicle state.
- Train or evaluate BP-NMPS and SI-RPPO directly on road unevenness sequences.
- Add preview noise, delay, scale error, and dropout to test robustness.
- This is the current thesis route and the easiest to defend.

### Stage B: Optional Chrono sensor dataset, future work

- Use Chrono::Sensor RGB/depth/lidar to render road observations.
- Store synchronized tuples:

```text
image/depth/lidar, clean road preview, noisy road preview, vehicle state, expert action, metrics
```

- Use fixed MPC-lite, BP-NMPS, and SI-RPPO as teachers.

### Stage C: Perception-to-preview model, future work

- Train a perception encoder to predict:
  - local elevation map,
  - preview height sequence,
  - roughness class,
  - preview confidence.
- Feed predicted preview and confidence into BP-NMPS or SI-RPPO gate.

### Stage D: VLA policy fine-tuning, future work

- Fine-tune a compact vision-language-action model or adapter.
- Prefer LoRA/adapters and frozen visual encoder.
- Output structured actions first: MPC gain residuals or gate parameters.
- Only test direct force output after safety projection and baseline fallback are stable.

## 6. Thesis Positioning

The safest thesis statement is:

> Chrono can be used as a high-fidelity vehicle dynamics and sensor backend for future VLA-style active suspension training. In the current thesis, the control contribution should remain BP-NMPS and SI-RPPO. Vision-based road perception can be treated as an upstream module that outputs preview sequences or confidence features.

The risky statement is:

> The thesis has already built a complete autonomous-driving VLA simulator comparable to SimplerEnv.

That would be too strong unless a full Chrono sensor pipeline, dataset, VLA policy, and sim-to-real validation are actually completed.

## 7. Best Near-Term Engineering Target

Build `SimpleSuspensionEnv`, a Gymnasium-compatible wrapper around the current full-car environment first. It should expose:

- observation dict: `state`, `history`, `road_preview`, `preview_confidence`, optional `image`;
- action dict: `mpc_gain_delta` or `force_residual`;
- task descriptor: comfort/safety/energy priority;
- expert dataset generation from SPDF, fixed MPC-lite, BP-NMPS, and SI-RPPO;
- evaluation table aligned with the existing thesis claim gates.

Chrono can be plugged in later as the rendering and dynamics backend once the control-side API is stable.
