# Phase 0: Single-Joint PD Validation Test

## Purpose
This test validates that PD control (Position + Velocity damping) eliminates oscillation before refactoring the entire robot control system.

**⚠️ CRITICAL**: If this test shows oscillation with PD control, DO NOT proceed with full implementation.

## Setup Instructions

### 1. Create Test Scene
1. Open Unity (version 6000.3.0f1)
2. Create new scene: `Assets/Scenes/Phase0_SingleJointTest.unity`
3. Add AR4 robot prefab to scene (or just a single joint)

### 2. Isolate Single Joint
1. Select the **first joint** (e.g., shoulder/base joint) of the AR4 robot
2. Disable all other joints (uncheck their GameObjects)
3. Ensure the selected joint has an `ArticulationBody` component

### 3. Attach Test Script
1. Add `SingleJointPDTest` component to the joint GameObject
2. Assign the joint's `ArticulationBody` to the script's `joint` field
3. Configure initial settings:
   - `useVelocityDamping`: **true** (PD control)
   - `positionGain (Kp)`: **10.0**
   - `velocityGain (Kd)`: **2.0**
   - `targetAngleDegrees`: **90.0**
   - `enableGraphing`: **true**

### 4. Run Test

#### Test 1: P-Only Control (Should Oscillate)
1. Set `useVelocityDamping = false`
2. Press Play
3. **Expected**: Joint oscillates around target, takes >2 seconds to settle

#### Test 2: PD Control (Should NOT Oscillate)
1. Set `useVelocityDamping = true`
2. Press Play
3. **Expected**: Joint smoothly approaches target, settles in <0.5 seconds

## Success Criteria

✅ **Phase 0 PASSES if:**
- P-only control (Kd=0) shows oscillation
- PD control (Kd=2) converges smoothly without oscillation
- Settling time with PD < 0.5 seconds
- Oscillation metric with PD < 0.01

❌ **Phase 0 FAILS if:**
- PD control still oscillates (check ArticulationBody stiffness/damping)
- Settling time > 1 second
- No visible difference between P-only and PD

## Runtime Controls

- **T**: Toggle between P-only and PD control
- **R**: Randomize target angle to re-test
- **Scene View**: See position graph above joint (cyan=PD, red=P-only)

## Interpretation

### Visual Graph
- **Green line**: Target position
- **Cyan line**: PD control trajectory (should be smooth S-curve)
- **Red line**: P-only trajectory (should show overshoot/oscillation)
- **Green sphere**: Convergence point (when settled)

### Console Output
```
[SingleJointPDTest] ✅ CONVERGED at t=0.42s (error=0.003°, vel=0.02 rad/s)
[SingleJointPDTest] Oscillation metric: 0.0087 (EXCELLENT)
```

## Troubleshooting

### Issue: PD control still oscillates
**Fix**:
1. Reduce `positionGain` (try 5.0)
2. Increase `velocityGain` (try 3.0-5.0)
3. Check ArticulationBody stiffness (should be ~2000, not 5000)

### Issue: Joint moves too slowly
**Fix**:
1. Increase `positionGain` (try 15-20)
2. Ensure ArticulationBody `forceLimit` is sufficient (>1000)

### Issue: No visible graph
**Fix**:
1. Ensure `enableGraphing = true`
2. Look above the joint in Scene view (not Game view)
3. Check Gizmos are enabled in Scene view

## Next Steps

**If Phase 0 PASSES**: Proceed to Phase 1 (full motion control implementation)

**If Phase 0 FAILS**:
1. Debug ArticulationBody configuration
2. Verify joint hierarchy is correct
3. Check for conflicting physics components
4. Consult plan document for ArticulationBody tuning guidance

## Reference

See full implementation plan: `ACRLPython/documents/RobotControlRedesign.md`
