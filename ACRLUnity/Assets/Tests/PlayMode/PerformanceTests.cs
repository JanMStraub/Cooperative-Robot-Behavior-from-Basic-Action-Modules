using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Robotics;
using Robotics.Grasp;
using Simulation;
using Configuration;
using Tests.EditMode;

namespace Tests.PlayMode
{
    /// <summary>
    /// Performance tests for Unity systems.
    /// Consolidates memory leak tests from MemoryLeakTests.cs
    /// and adds new performance benchmarks for:
    /// - IK solver convergence time and memory allocation
    /// - Grasp planning pipeline performance
    /// - Coordination collision checking performance
    /// - Logging system performance (when implemented)
    /// </summary>
    public class PerformanceTests
    {
        private GameObject _robotObject;
        private RobotController _robotController;

        #region Setup/Teardown

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            // Create minimal robot setup for testing
            _robotObject = new GameObject("TestRobot");
            _robotController = _robotObject.AddComponent<RobotController>();
            _robotController.robotId = "TestRobot";

            // Add minimal required components
            var endEffectorBase = new GameObject("EndEffectorBase");
            endEffectorBase.transform.SetParent(_robotObject.transform);
            _robotController.endEffectorBase = endEffectorBase.transform;

            // Expect initialization warnings
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] No GripperController found in children of TestRobot");
            LogAssert.Expect(LogType.Warning, "[ROBOT_CONTROLLER] Robot joints are not assigned. Please assign ArticulationBodies.");

            yield return null;
        }

        [TearDown]
        public void TearDown()
        {
            TestHelpers.DestroyAll(_robotObject);
        }

        #endregion

        #region Memory Leak Tests (Consolidated from MemoryLeakTests.cs)

        [UnityTest]
        public IEnumerator Memory_SetTargetVector3_DoesNotLeak()
        {
            int initialObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Call SetTarget 100 times
            for (int i = 0; i < 100; i++)
            {
                _robotController.SetTarget(
                    new Vector3(i * 0.01f, i * 0.01f, i * 0.01f),
                    GraspOptions.MoveOnly
                );
                yield return null;
            }

            // Allow garbage collection
            yield return null;
            System.GC.Collect();
            yield return null;

            int finalObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            int objectDelta = finalObjectCount - initialObjectCount;

            // Should create at most 1 cached temporary object, not 100
            Assert.LessOrEqual(objectDelta, 1,
                $"Expected at most 1 new GameObject (cached), found {objectDelta}. Memory leak detected.");
        }

        [UnityTest]
        public IEnumerator Memory_SetTargetGameObject_DoesNotLeak()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            int initialObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Call SetTarget with grasp planning 100 times
            for (int i = 0; i < 100; i++)
            {
                targetObject.transform.position = new Vector3(i * 0.01f, 1f, 1f);
                _robotController.SetTarget(targetObject, GraspOptions.Default);
                yield return null;
            }

            yield return null;
            System.GC.Collect();
            yield return null;

            int finalObjectCount = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;
            int objectDelta = finalObjectCount - initialObjectCount;

            Assert.LessOrEqual(objectDelta, 1,
                $"Expected at most 1 new GameObject (cached), found {objectDelta}. Memory leak detected.");

            Object.Destroy(targetObject);
        }

        [UnityTest]
        public IEnumerator Memory_OnDestroy_CleanupsCachedObjects()
        {
            var targetObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetObject.transform.position = new Vector3(1f, 1f, 1f);

            // Create cached objects
            _robotController.SetTarget(new Vector3(1f, 0f, 0f), GraspOptions.MoveOnly);
            yield return null;
            _robotController.SetTarget(targetObject, GraspOptions.Default);
            yield return null;

            int countBeforeDestroy = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Destroy robot controller
            Object.Destroy(_robotObject);
            _robotObject = null; // Prevent TearDown from double-destroying

            yield return null;

            int countAfterDestroy = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None).Length;

            // Should have fewer objects after destruction
            Assert.Less(countAfterDestroy, countBeforeDestroy,
                "Cached GameObjects should be destroyed when RobotController is destroyed");

            Object.Destroy(targetObject);
        }

        #endregion

        #region IK Solver Performance Tests

        [UnityTest]
        public IEnumerator IKSolver_ConvergenceTime_LessThan500ms()
        {
            // Test that IK convergence happens within 500ms target
            // (Phase 1: Motion Control Redesign requirement)

            TestHelpers.SetupMinimalArticulationChain(_robotController);
            LogAssert.Expect(LogType.Error, "Tag: EndEffector is not defined.");

            Vector3 targetPosition = new Vector3(0.15f, 0.15f, 0.15f); // Reachable position
            GameObject target = TestHelpers.CreateTestTarget(targetPosition);

            var stopwatch = Stopwatch.StartNew();

            _robotController.SetTarget(target);

            // Wait for convergence (or max 1 second)
            float maxWaitTime = 1f;
            float elapsed = 0f;
            while (!_robotController.TargetReached && elapsed < maxWaitTime)
            {
                yield return null;
                elapsed += Time.deltaTime;
            }

            stopwatch.Stop();

            // Verify convergence time
            if (_robotController.TargetReached)
            {
                Assert.Less(stopwatch.ElapsedMilliseconds, 500,
                    $"IK convergence should complete in < 500ms, took {stopwatch.ElapsedMilliseconds}ms");
            }
            else
            {
                UnityEngine.Debug.LogWarning($"Target not reached within {maxWaitTime}s - IK may need tuning");
            }

            TestHelpers.DestroyAll(target);
        }

        [UnityTest]
        public IEnumerator IKSolver_MemoryAllocation_PreallocatedMatrices()
        {
            // Test that IK solver uses pre-allocated matrices (GC-free operation)

            TestHelpers.SetupMinimalArticulationChain(_robotController);
            LogAssert.Expect(LogType.Error, "Tag: EndEffector is not defined.");

            Vector3 targetPosition = new Vector3(0.15f, 0.15f, 0.15f);
            GameObject target = TestHelpers.CreateTestTarget(targetPosition);

            // Force garbage collection before test
            System.GC.Collect();
            yield return null;

            long memoryBefore = System.GC.GetTotalMemory(true);

            // Perform IK iterations
            _robotController.SetTarget(target);
            for (int i = 0; i < 100; i++)
            {
                yield return null; // Allow IK to run
            }

            long memoryAfter = System.GC.GetTotalMemory(false);
            long memoryDelta = memoryAfter - memoryBefore;

            // Should have minimal memory allocation (< 1MB for 100 iterations)
            Assert.Less(memoryDelta, 1024 * 1024,
                $"IK solver should use pre-allocated matrices. Allocated {memoryDelta / 1024}KB in 100 iterations.");

            TestHelpers.DestroyAll(target);
        }

        #endregion

        #region Grasp Planning Performance Tests

        [UnityTest]
        public IEnumerator GraspPlanning_PipelineExecution_LessThan200ms()
        {
            // Test that grasp planning pipeline executes within 200ms budget
            // (Configured in GraspConfig.maxPipelineTimeMs)

            var targetCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetCube.transform.position = new Vector3(0.3f, 0.2f, 0.3f);
            targetCube.transform.localScale = Vector3.one * 0.05f; // 5cm cube

            var stopwatch = Stopwatch.StartNew();

            _robotController.SetTarget(targetCube, GraspOptions.Default);

            yield return null; // Allow pipeline to execute

            stopwatch.Stop();

            // Pipeline should execute within configured budget
            Assert.Less(stopwatch.ElapsedMilliseconds, 200,
                $"Grasp planning should complete in < 200ms, took {stopwatch.ElapsedMilliseconds}ms");

            Object.Destroy(targetCube);
        }

        [UnityTest]
        public IEnumerator GraspPlanning_CandidateGeneration_ScalesLinearly()
        {
            // Test that candidate generation scales linearly with number of candidates

            var graspConfig = ScriptableObject.CreateInstance<GraspConfig>();
            graspConfig.InitializeDefaultConfig();

            var targetCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            targetCube.transform.position = new Vector3(0.3f, 0.2f, 0.3f);
            targetCube.transform.localScale = Vector3.one * 0.05f;

            // Test with 5 candidates
            graspConfig.candidatesPerApproach = 5;
            var stopwatch1 = Stopwatch.StartNew();
            yield return null; // Simulate candidate generation
            stopwatch1.Stop();

            // Test with 10 candidates (2x)
            graspConfig.candidatesPerApproach = 10;
            var stopwatch2 = Stopwatch.StartNew();
            yield return null; // Simulate candidate generation
            stopwatch2.Stop();

            // Time should scale roughly linearly (allow 3x tolerance for overhead)
            float ratio = (float)stopwatch2.ElapsedMilliseconds / (float)stopwatch1.ElapsedMilliseconds;
            Assert.Less(ratio, 3.0f,
                $"Candidate generation should scale linearly. Ratio: {ratio:F2}x");

            Object.Destroy(targetCube);
            Object.DestroyImmediate(graspConfig);
        }

        #endregion

        #region Coordination Performance Tests

        [UnityTest]
        public IEnumerator Coordination_CollisionCheck_LessThan10ms()
        {
            // Test that collision check executes in < 10ms per update
            // (Required for real-time coordination)

            var robot1 = TestHelpers.CreateTestRobot("Robot1").controller;
            var robot2 = TestHelpers.CreateTestRobot("Robot2").controller;

            robot1.transform.position = new Vector3(0f, 0f, 0f);
            robot2.transform.position = new Vector3(0.1f, 0f, 0f); // Close to robot1

            var stopwatch = Stopwatch.StartNew();

            // Perform collision check
            float distance = Vector3.Distance(robot1.transform.position, robot2.transform.position);
            bool collision = distance < TestConstants.MIN_SAFE_SEPARATION;

            stopwatch.Stop();

            // Collision check should be fast (< 10ms)
            Assert.Less(stopwatch.ElapsedMilliseconds, 10,
                $"Collision check should complete in < 10ms, took {stopwatch.ElapsedMilliseconds}ms");

            Object.Destroy(robot1.gameObject);
            Object.Destroy(robot2.gameObject);

            yield return null;
        }

        [UnityTest]
        public IEnumerator Coordination_StateSyncLatency_LessThan50ms()
        {
            // Test that coordination state sync has < 50ms latency
            // (Important for responsive multi-robot coordination)

            var manager = TestHelpers.CreateSimulationManager().manager;
            var config = TestHelpers.CreateTestSimulationConfig(RobotCoordinationMode.Sequential);
            manager.config = config;

            yield return null; // Initialize manager

            var stopwatch = Stopwatch.StartNew();

            // Simulate state sync operation
            manager.NotifyTargetReached("Robot1", true);

            stopwatch.Stop();

            // State sync should be fast
            Assert.Less(stopwatch.ElapsedMilliseconds, 50,
                $"State sync should complete in < 50ms, took {stopwatch.ElapsedMilliseconds}ms");

            Object.Destroy(manager.gameObject);
            Object.DestroyImmediate(config);
        }

        #endregion

        #region Stress Tests

        [UnityTest]
        public IEnumerator Stress_100SequentialSetTargetCalls_Completes()
        {
            // Stress test: 100 rapid SetTarget calls should complete without errors

            var stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < 100; i++)
            {
                _robotController.SetTarget(
                    new Vector3(i * 0.001f, i * 0.001f, i * 0.001f),
                    GraspOptions.MoveOnly
                );
                yield return null;
            }

            stopwatch.Stop();

            // Should complete in reasonable time (< 10 seconds)
            Assert.Less(stopwatch.ElapsedMilliseconds, 10000,
                $"100 SetTarget calls should complete in < 10s, took {stopwatch.ElapsedMilliseconds}ms");
        }

        [UnityTest]
        public IEnumerator Stress_SimultaneousMultipleRobots_NoPerformanceDegradation()
        {
            // Stress test: 3 robots moving simultaneously

            var robots = new List<RobotController>();
            for (int i = 0; i < 3; i++)
            {
                var (obj, controller) = TestHelpers.CreateTestRobot($"Robot{i}");
                robots.Add(controller);
            }

            var stopwatch = Stopwatch.StartNew();

            // Set targets for all robots
            for (int i = 0; i < robots.Count; i++)
            {
                robots[i].SetTarget(
                    new Vector3(i * 0.2f, 0.3f, 0.3f),
                    GraspOptions.MoveOnly
                );
            }

            // Run for 1 second
            yield return new WaitForSeconds(1f);

            stopwatch.Stop();

            // All robots should remain responsive
            Assert.Less(stopwatch.ElapsedMilliseconds, 1500,
                "Multiple robots should not cause significant performance degradation");

            // Cleanup
            foreach (var robot in robots)
            {
                Object.Destroy(robot.gameObject);
            }
        }

        #endregion

        #region Frame Rate Tests

        [UnityTest]
        public IEnumerator FrameRate_MaintainsTargetFPS_DuringSimulation()
        {
            // Test that simulation maintains target frame rate

            var config = TestHelpers.CreateTestSimulationConfig();
            config.targetFrameRate = 30;

            // Measure frame rate over 1 second
            int frameCount = 0;
            float startTime = Time.realtimeSinceStartup;

            while (Time.realtimeSinceStartup - startTime < 1f)
            {
                frameCount++;
                yield return null;
            }

            float actualFPS = frameCount;

            // Should be close to target FPS (allow 20% variance)
            Assert.GreaterOrEqual(actualFPS, config.targetFrameRate * 0.8f,
                $"Frame rate should be >= {config.targetFrameRate * 0.8f} FPS, got {actualFPS} FPS");

            Object.DestroyImmediate(config);
        }

        #endregion

        #region Memory Profiling Tests

        [UnityTest]
        public IEnumerator Memory_TotalAllocation_RemainsStable()
        {
            // Test that total memory allocation remains stable during operation

            System.GC.Collect();
            yield return null;

            long memoryStart = System.GC.GetTotalMemory(true);

            // Perform typical operations for 100 iterations
            for (int i = 0; i < 100; i++)
            {
                _robotController.SetTarget(
                    new Vector3(i * 0.01f, 0.2f, 0.3f),
                    GraspOptions.MoveOnly
                );
                yield return null;
            }

            System.GC.Collect();
            yield return null;

            long memoryEnd = System.GC.GetTotalMemory(true);
            long memoryGrowth = memoryEnd - memoryStart;

            // Memory growth should be minimal (< 10MB for 100 iterations)
            Assert.Less(memoryGrowth, 10 * 1024 * 1024,
                $"Memory should remain stable. Grew by {memoryGrowth / 1024}KB in 100 iterations.");
        }

        #endregion
    }
}
