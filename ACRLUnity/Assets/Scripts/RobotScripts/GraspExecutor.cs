using System;
using System.Collections;
using Configuration;
using Core;
using Robotics.Grasp;
using UnityEngine;

namespace Robotics
{
    /// <summary>Grasp coroutines extracted from RobotController; borrows coroutine execution via injected owner.</summary>
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
        private readonly GraspConfig _graspTimingConfig;
        private readonly SimpleRobotController _simpleRobotController;
        private readonly string _robotId;
        private readonly string _logPrefix;

        // GraspConfig values with safe defaults
        private float GraspTimeoutFallback =>
            _graspTimingConfig != null ? _graspTimingConfig.graspTimeoutFallbackSeconds : 30f;
        private float VelocityThresholdPostPreGrasp =>
            _graspTimingConfig != null ? _graspTimingConfig.velocityThresholdPostPreGrasp : 0.01f;
        private float VelocityThresholdPreGripperClose =>
            _graspTimingConfig != null
                ? _graspTimingConfig.velocityThresholdPreGripperClose
                : 0.005f;
        private float GraspConfirmationWait =>
            _graspTimingConfig != null ? _graspTimingConfig.graspConfirmationWaitSeconds : 0.3f;

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
            Action<Coroutine> setActiveCoroutine,
            GraspConfig graspTimingConfig = null
        )
        {
            _owner = owner;
            _gripperController = gripperController;
            _ikConfig = ikConfig;
            _graspTimingConfig = graspTimingConfig;
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

        public IEnumerator WaitForTargetWithTimeout(
            Func<bool> hasReachedTarget,
            float timeoutSeconds
        )
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
        /// Waits for the robot to reach the target (with timeout), then waits for end-effector
        /// velocity to drop below the given threshold. Invokes onTimeout if the timeout expires
        /// before the target is reached.
        /// </summary>
        private IEnumerator WaitForGraspPhase(
            Func<bool> hasReachedTarget,
            float timeoutSeconds,
            float velocityThreshold,
            System.Action onTimeout = null)
        {
            yield return _owner.StartCoroutine(
                WaitForTargetWithTimeout(hasReachedTarget, timeoutSeconds)
            );

            if (!hasReachedTarget())
            {
                onTimeout?.Invoke();
                yield break;
            }

            yield return new WaitUntil(() =>
                _getEndEffectorVelocityMagnitude() < velocityThreshold
            );
        }

        public IEnumerator CloseGripperAfterDelay(
            GameObject targetObject,
            float gripperCloseDelay,
            bool attachObjectOnGrasp
        )
        {
            float delayStartTime = Time.time;
            yield return new WaitUntil(() =>
                Time.time - delayStartTime >= gripperCloseDelay
                && _getEndEffectorVelocityMagnitude() < 0.005f
            );

            if (attachObjectOnGrasp && targetObject != null)
                _gripperController.SetTargetObject(targetObject);

            _gripperController.CloseGrippers();
            yield return new WaitWhile(() => _gripperController.IsMoving);

            float graspStartTime = Time.time;
            yield return new WaitUntil(() =>
                Time.time - graspStartTime > 0.2f && !_gripperController.IsMoving
            );

            _fireOnTargetReached();
        }

        /// <summary>Two-waypoint grasp: pre-grasp → grasp position.</summary>
        public IEnumerator ExecuteTwoWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options,
            Func<bool> hasReachedTarget
        )
        {
            float graspTimeout =
                _ikConfig != null ? _ikConfig.graspTimeoutSeconds : GraspTimeoutFallback;

            _gripperController?.SetGripperPosition(candidate.preGraspGripperWidth);

            GameObject pre = _getCachedTempObject("_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );

            _setIsGraspingTarget(false);
            _setTargetInternal(
                pre.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return _owner.StartCoroutine(
                WaitForGraspPhase(hasReachedTarget, graspTimeout, VelocityThresholdPostPreGrasp)
            );

            GameObject main = _getCachedTempObject(RobotConstants.GRASP_TARGET_SUFFIX);
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _setIsGraspingTarget(true);
            _setTargetInternal(
                main.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return _owner.StartCoroutine(
                WaitForTargetWithTimeout(hasReachedTarget, graspTimeout)
            );

            if (!hasReachedTarget())
                yield break;

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitUntil(() =>
                    _getEndEffectorVelocityMagnitude() < VelocityThresholdPreGripperClose
                );
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(() =>
                    Time.time - graspStartTime > GraspConfirmationWait
                    && !_gripperController.IsMoving
                );
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }

        /// <summary>Three-waypoint grasp: pre-grasp → grasp → optional retreat.</summary>
        public IEnumerator ExecuteThreeWaypointGrasp(
            GraspCandidate candidate,
            GameObject targetObject,
            GraspOptions options,
            Func<bool> hasReachedTarget
        )
        {
            float graspTimeout =
                _ikConfig != null ? _ikConfig.graspTimeoutSeconds : GraspTimeoutFallback;

            _gripperController?.SetGripperPosition(candidate.preGraspGripperWidth);

            GameObject pre = _getCachedTempObject("_pre");
            pre.transform.SetPositionAndRotation(
                candidate.preGraspPosition,
                candidate.preGraspRotation
            );

            _setIsGraspingTarget(false);
            _setTargetInternal(
                pre.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return _owner.StartCoroutine(
                WaitForGraspPhase(hasReachedTarget, graspTimeout, VelocityThresholdPostPreGrasp)
            );

            GameObject main = _getCachedTempObject(RobotConstants.GRASP_TARGET_SUFFIX);
            main.transform.SetPositionAndRotation(candidate.graspPosition, candidate.graspRotation);

            _setIsGraspingTarget(true);
            _setTargetInternal(
                main.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );
            yield return _owner.StartCoroutine(
                WaitForTargetWithTimeout(hasReachedTarget, graspTimeout)
            );

            if (!hasReachedTarget())
                yield break;

            if (options.closeGripperOnReach && _gripperController != null)
            {
                yield return new WaitUntil(() =>
                    _getEndEffectorVelocityMagnitude() < VelocityThresholdPreGripperClose
                );
                _gripperController.SetTargetObject(targetObject);
                _gripperController.SetGripperPosition(candidate.graspGripperWidth);
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(() =>
                    Time.time - graspStartTime > GraspConfirmationWait
                    && !_gripperController.IsMoving
                );
            }

            // Retreat
            if (options.graspConfig != null && options.graspConfig.enableRetreat)
            {
                GameObject retreat = _getCachedTempObject("_retreat");
                retreat.transform.SetPositionAndRotation(
                    candidate.retreatPosition,
                    candidate.retreatRotation
                );

                _setIsGraspingTarget(false);
                _setTargetInternal(
                    retreat.transform,
                    targetObject,
                    new GraspOptions { closeGripperOnReach = false }
                );
                yield return _owner.StartCoroutine(
                    WaitForTargetWithTimeout(hasReachedTarget, graspTimeout)
                );
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }

        /// <summary>Handoff grasp: take object from another robot's gripper.</summary>
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
            _setTargetInternal(
                handoffTarget.transform,
                targetObject,
                new GraspOptions { closeGripperOnReach = false }
            );

            float graspTimeout =
                _ikConfig != null ? _ikConfig.graspTimeoutSeconds : GraspTimeoutFallback;
            yield return _owner.StartCoroutine(
                WaitForGraspPhase(
                    hasReachedTarget,
                    graspTimeout,
                    VelocityThresholdPreGripperClose,
                    onTimeout: () => {
                        Debug.LogWarning($"{_logPrefix} {_robotId} failed to reach handoff position");
                        _setActiveCoroutine(null);
                    }
                )
            );

            if (options.closeGripperOnReach && _gripperController != null)
            {
                _gripperController.SetTargetObject(targetObject);
                _gripperController.CloseGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);

                float graspStartTime = Time.time;
                yield return new WaitUntil(() =>
                    Time.time - graspStartTime > GraspConfirmationWait
                    && !_gripperController.IsMoving
                );
            }

            _setActiveCoroutine(null);
            _fireOnTargetReached();
        }

        /// <summary>Fallback grasp via SimpleRobotController IK when planning fails.</summary>
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
                    yield return _owner.StartCoroutine(
                        ExecuteTwoWaypointGrasp(candidate, targetObject, options, hasReachedTarget)
                    );
                yield break;
            }

            if (options.openGripperOnSet && _gripperController != null)
            {
                _gripperController.OpenGrippers();
                yield return new WaitWhile(() => _gripperController.IsMoving);
            }

            _simpleRobotController.SetTarget(candidate.graspPosition, candidate.graspRotation);

            float timeout =
                _ikConfig != null ? _ikConfig.graspTimeoutSeconds : GraspTimeoutFallback;
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
                yield return new WaitUntil(() =>
                    Time.time - graspStartTime > GraspConfirmationWait
                    && !_gripperController.IsMoving
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
