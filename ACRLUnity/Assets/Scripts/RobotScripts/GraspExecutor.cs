using System;
using System.Collections;
using Configuration;
using Core;
using Robotics.Grasp;
using UnityEngine;

namespace Robotics
{
    /// <summary>
    /// Encapsulates all grasp coroutine logic extracted from RobotController.
    ///
    /// GraspExecutor is a plain class (not MonoBehaviour) that borrows coroutine
    /// execution from its owner RobotController. It receives state access via
    /// constructor-injected callbacks and direct references, keeping RobotController
    /// focused on IK orchestration and physics.
    /// </summary>
    public class GraspExecutor
    {
        // Callbacks into RobotController state
        private readonly Action<Transform, GameObject, GraspOptions> _setTargetInternal;
        private readonly Func<float> _getEndEffectorVelocityMagnitude;
        private readonly Func<string, GameObject> _getCachedTempObject;
        private readonly Action<bool> _setIsGraspingTarget;
        private readonly Action _fireOnTargetReached;
        private readonly Action<Coroutine> _setActiveCoroutine;

        // Direct references
        private readonly MonoBehaviour _owner;
        private readonly GripperController _gripperController;
        private readonly IKConfig _ikConfig;
        private readonly SimpleRobotController _simpleRobotController;
        private readonly string _robotId;
        private readonly string _logPrefix;

        /// <summary>
        /// Construct a GraspExecutor with all required dependencies injected.
        /// </summary>
        /// <param name="owner">MonoBehaviour that hosts coroutine execution (typically RobotController)</param>
        /// <param name="gripperController">Gripper component for open/close operations</param>
        /// <param name="ikConfig">IK configuration for timeouts and thresholds</param>
        /// <param name="simpleRobotController">Fallback IK controller for simplified grasps</param>
        /// <param name="robotId">Robot identifier for log messages</param>
        /// <param name="logPrefix">Log prefix string</param>
        /// <param name="setTargetInternal">Delegate to RobotController.SetTargetInternal</param>
        /// <param name="getEndEffectorVelocityMagnitude">Delegate returning current EE velocity magnitude</param>
        /// <param name="getCachedTempObject">Delegate returning a cached temp GameObject by suffix</param>
        /// <param name="setIsGraspingTarget">Delegate to set RobotController._isGraspingTarget</param>
        /// <param name="fireOnTargetReached">Delegate to invoke RobotController.OnTargetReached</param>
        /// <param name="setActiveCoroutine">Delegate to update RobotController._activeGraspCoroutine</param>
        public GraspExecutor(
            MonoBehaviour owner,
            GripperController gripperController,
            IKConfig ikConfig,
            SimpleRobotController simpleRobotController,
            string robotId,
            string logPrefix,
            Action<Transform, GameObject, GraspOptions> setTargetInternal,
            Func<float> getEndEffectorVelocityMagnitude,
            Func<string, GameObject> getCachedTempObject,
            Action<bool> setIsGraspingTarget,
            Action fireOnTargetReached,
            Action<Coroutine> setActiveCoroutine
        )
        {
            _owner = owner;
            _gripperController = gripperController;
            _ikConfig = ikConfig;
            _simpleRobotController = simpleRobotController;
            _robotId = robotId;
            _logPrefix = logPrefix;
            _setTargetInternal = setTargetInternal;
            _getEndEffectorVelocityMagnitude = getEndEffectorVelocityMagnitude;
            _getCachedTempObject = getCachedTempObject;
            _setIsGraspingTarget = setIsGraspingTarget;
            _fireOnTargetReached = fireOnTargetReached;
            _setActiveCoroutine = setActiveCoroutine;
        }

        /// <summary>
        /// Coroutine that waits until HasReachedTarget is true or timeout elapses.
        /// </summary>
        /// <param name="hasReachedTarget">Getter returning current reach state</param>
        /// <param name="timeoutSeconds">Maximum seconds to wait</param>
        public IEnumerator WaitForTargetWithTimeout(Func<bool> hasReachedTarget, float timeoutSeconds)
        {
            float startTime = Time.time;
            while (!hasReachedTarget())
            {
                if (Time.time - startTime > timeoutSeconds)
                {
                    Debug.LogWarning($"{_logPrefix} {_robotId} timeout after {timeoutSeconds}s");
                    yield break;
                }
                yield return null;
            }
        }

        /// <summary>
        /// Coroutine that delays gripper close until velocity settles, then attaches and
        /// closes the gripper before invoking OnTargetReached.
        /// </summary>
        /// <param name="targetObject">Object to attach on grasp</param>
        /// <param name="gripperCloseDelay">Seconds to wait before closing</param>
        /// <param name="attachObjectOnGrasp">Whether to call SetTargetObject before closing</param>
        public IEnumerator CloseGripperAfterDelay(
            GameObject targetObject,
            float gripperCloseDelay,
            bool attachObjectOnGrasp
        )
        {
            float delayStartTime = Time.time;
            yield return new WaitUntil(
                () =>
                    Time.time - delayStartTime >= gripperCloseDelay
                    && _getEndEffectorVelocityMagnitude() < 0.005f
            );

            if (attachObjectOnGrasp && targetObject != null)
                _gripperController.SetTargetObject(targetObject);

            _gripperController.CloseGrippers();
            yield return new WaitWhile(() => _gripperController.IsMoving);

            float graspStartTime = Time.time;
            yield return new WaitUntil(
                () => Time.time - graspStartTime > 0.2f && !_gripperController.IsMoving
            );

            _fireOnTargetReached();
        }

        /// <summary>
        /// Two-waypoint grasp: pre-grasp position → grasp position (optional gripper close).
        /// </summary>
        /// <param name="candidate">Planned grasp candidate with pre/grasp positions</param>
        /// <param name="targetObject">Object being grasped</param>
        /// <param name="options">Grasp options</param>
        /// <param name="hasReachedTarget">Getter for current reach state</param>
        public IEnumerator ExecuteTwoWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options,
            Func<bool> hasReachedTarget
        )
        {
            float graspTimeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;

            _gripperController?.SetGripperPosition(candidate.preGraspGripperWidth);

            // 1. Pre-Grasp
            GameObject pre = _getCachedTempObject("_pre");
            pre.transform.SetPositionAndRotation(candidate.preGraspPosition, candidate.preGraspRotation);

            _setIsGraspingTarget(false);
            _setTargetInternal(pre.transform, targetObject, new GraspOptions { closeGripperOnReach = false });
            yield return _owner.StartCoroutine(WaitForTargetWithTimeout(hasReachedTarget, graspTimeout));

            if (!hasReachedTarget())
                yield break;

            yield return new WaitUntil(() => _getEndEffectorVelocityMagnitude() < 0.01f);

            // 2. Grasp
            GameObject main = _getCachedTempObject(RobotConstants.GRASP_TARGET_SUFFIX);
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _setIsGraspingTarget(true);
            _setTargetInternal(main.transform, targetObject, new GraspOptions { closeGripperOnReach = false });
            yield return _owner.StartCoroutine(WaitForTargetWithTimeout(hasReachedTarget, graspTimeout));

            if (!hasReachedTarget())
                yield break;

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitUntil(() => _getEndEffectorVelocityMagnitude() < 0.005f);
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }

        /// <summary>
        /// Three-waypoint grasp: pre-grasp → grasp → optional retreat.
        /// </summary>
        /// <param name="candidate">Planned grasp candidate with pre/grasp/retreat positions</param>
        /// <param name="targetObject">Object being grasped</param>
        /// <param name="options">Grasp options</param>
        /// <param name="hasReachedTarget">Getter for current reach state</param>
        public IEnumerator ExecuteThreeWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options,
            Func<bool> hasReachedTarget
        )
        {
            float graspTimeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;

            _gripperController?.SetGripperPosition(candidate.preGraspGripperWidth);

            // 1. Pre
            GameObject pre = _getCachedTempObject("_pre");
            pre.transform.SetPositionAndRotation(candidate.preGraspPosition, candidate.preGraspRotation);

            _setIsGraspingTarget(false);
            _setTargetInternal(pre.transform, targetObject, new GraspOptions { closeGripperOnReach = false });
            yield return _owner.StartCoroutine(WaitForTargetWithTimeout(hasReachedTarget, graspTimeout));

            if (!hasReachedTarget())
                yield break;
            yield return new WaitUntil(() => _getEndEffectorVelocityMagnitude() < 0.01f);

            // 2. Grasp
            GameObject main = _getCachedTempObject(RobotConstants.GRASP_TARGET_SUFFIX);
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _setIsGraspingTarget(true);
            _setTargetInternal(main.transform, targetObject, new GraspOptions { closeGripperOnReach = false });
            yield return _owner.StartCoroutine(WaitForTargetWithTimeout(hasReachedTarget, graspTimeout));

            if (!hasReachedTarget())
                yield break;

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitUntil(() => _getEndEffectorVelocityMagnitude() < 0.005f);
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );
            }

            // 3. Retreat
            if (options.graspConfig != null && options.graspConfig.enableRetreat)
            {
                GameObject retreat = _getCachedTempObject("_retreat");
                retreat.transform.SetPositionAndRotation(candidate.retreatPosition, candidate.retreatRotation);

                _setIsGraspingTarget(false);
                _setTargetInternal(retreat.transform, targetObject, new GraspOptions { closeGripperOnReach = false });
                yield return _owner.StartCoroutine(WaitForTargetWithTimeout(hasReachedTarget, graspTimeout));
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }

        /// <summary>
        /// Handoff grasp: receives an object held by another robot's gripper.
        /// Opens gripper, moves to handoff position, then optionally closes.
        /// </summary>
        /// <param name="targetObject">Object to take via handoff</param>
        /// <param name="options">Grasp options</param>
        /// <param name="hasReachedTarget">Getter for current reach state</param>
        public IEnumerator ExecuteHandoffGrasp(
            GameObject targetObject,
            GraspOptions options,
            Func<bool> hasReachedTarget
        )
        {
            Debug.Log($"{_logPrefix} {_robotId} executing handoff for '{targetObject.name}'");

            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);
            }

            Vector3 objectPosition = targetObject.transform.position;
            GameObject handoffTarget = _getCachedTempObject("_handoff");
            handoffTarget.transform.position = objectPosition;
            handoffTarget.transform.rotation = targetObject.transform.rotation;

            _setIsGraspingTarget(true);
            _setTargetInternal(handoffTarget.transform, targetObject, new GraspOptions { closeGripperOnReach = false });

            float graspTimeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;
            yield return _owner.StartCoroutine(WaitForTargetWithTimeout(hasReachedTarget, graspTimeout));

            if (!hasReachedTarget())
            {
                Debug.LogWarning($"{_logPrefix} {_robotId} failed to reach handoff position");
                _setActiveCoroutine(null);
                yield break;
            }

            yield return new WaitUntil(() => _getEndEffectorVelocityMagnitude() < 0.005f);

            if (options.closeGripperOnReach && _gripperController != null)
            {
                _gripperController.SetTargetObject(targetObject);
                _gripperController.CloseGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }

        /// <summary>
        /// Simplified grasp using SimpleRobotController's IK as fallback when advanced
        /// grasp planning fails. Manually steps the SimpleRobotController each physics frame.
        /// </summary>
        /// <param name="candidate">Grasp candidate from fallback planner</param>
        /// <param name="targetObject">Object to grasp</param>
        /// <param name="options">Grasp options</param>
        /// <param name="hasReachedTarget">Getter for current reach state (used in two-waypoint fallback)</param>
        public IEnumerator ExecuteSimplifiedGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options,
            Func<bool> hasReachedTarget = null
        )
        {
            Debug.Log(
                $"{_logPrefix} {_robotId} executing SIMPLIFIED grasp using SimpleRobotController backup IK (fallback mode)"
            );

            if (_simpleRobotController == null)
            {
                Debug.LogWarning(
                    $"{_logPrefix} {_robotId} SimpleRobotController not assigned! Falling back to two-waypoint execution."
                );
                if (hasReachedTarget != null)
                    yield return _owner.StartCoroutine(ExecuteTwoWaypointGrasp(candidate, targetObject, options, hasReachedTarget));
                yield break;
            }

            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);
            }

            _simpleRobotController.SetTarget(candidate.graspPosition, candidate.graspRotation);

            float timeout = _ikConfig != null ? _ikConfig.graspTimeoutSeconds : 30f;
            float startTime = Time.time;

            while (!_simpleRobotController.HasReachedTarget)
            {
                if (Time.time - startTime > timeout)
                {
                    Debug.LogWarning(
                        $"{_logPrefix} {_robotId} simplified grasp timed out after {timeout}s"
                    );
                    _setActiveCoroutine(null);
                    yield break;
                }
                _simpleRobotController.PerformInverseKinematicsStep();
                yield return new WaitForFixedUpdate();
            }

            Debug.Log(
                $"{_logPrefix} {_robotId} simplified grasp reached target position. Distance: {_simpleRobotController.DistanceToTarget:F4}m"
            );

            yield return new WaitForSeconds(0.2f);

            if (options.closeGripperOnReach && _gripperController != null)
            {
                _gripperController.SetTargetObject(targetObject);
                _gripperController.CloseGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(
                    () => Time.time - graspStartTime > 0.3f && !_gripperController.IsMoving
                );

                Debug.Log(
                    $"{_logPrefix} {_robotId} simplified grasp complete. Object held: {_gripperController.IsHoldingObject}"
                );
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }
    }
}
