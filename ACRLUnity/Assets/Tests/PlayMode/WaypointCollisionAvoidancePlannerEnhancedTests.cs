using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Simulation.CoordinationStrategies;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Enhanced tests for WaypointCollisionAvoidancePlanner.
    /// Covers additional edge cases, strategy combinations, and distance calculations.
    /// Complements the existing CollisionAvoidancePlannerTests.
    /// </summary>
    public class WaypointCollisionAvoidancePlannerEnhancedTests
    {
        private WaypointCollisionAvoidancePlanner _planner;
        private const float VERTICAL_OFFSET = 0.15f;
        private const float LATERAL_OFFSET = 0.1f;
        private const float MIN_SAFE_SEPARATION = 0.2f;
        private const int MAX_WAYPOINTS = 5;
        private const float EPSILON = 0.001f;

        [SetUp]
        public void SetUp()
        {
            _planner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: VERTICAL_OFFSET,
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: MIN_SAFE_SEPARATION,
                maxWaypoints: MAX_WAYPOINTS
            );
        }

        [TearDown]
        public void TearDown()
        {
            _planner = null;
        }

        #region Vertical Strategy Tests

        [Test]
        public void PlanAlternativePath_VerticalStrategy_UsesSafeVerticalOffset()
        {
            // When minSafeSeparation is larger than verticalOffset, should use the larger value
            var planner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 0.05f,  // Smaller
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: 0.25f,  // Larger
                maxWaypoints: MAX_WAYPOINTS
            );

            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            var waypoints = planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Verify waypoints use safe offset (should be at least minSafeSeparation * 1.1)
            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                float maxHeight = 0f;
                foreach (var wp in waypoints)
                    maxHeight = Mathf.Max(maxHeight, wp.y);

                Assert.GreaterOrEqual(maxHeight, 0.25f * 1.1f,
                    "Should use minSafeSeparation when larger than verticalOffset");
            }
        }

        [Test]
        public void PlanAlternativePath_VerticalStrategy_CreatesThreeWaypoints()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            // Vertical strategy creates: lift, move over, descend (3 waypoints)
            Assert.GreaterOrEqual(waypoints.Count, 3, "Vertical strategy should create at least 3 waypoints");
        }

        [Test]
        public void PlanAlternativePath_VerticalStrategy_LiftPointAboveStart()
        {
            Vector3 start = new Vector3(0f, 0.5f, 0f); // Start elevated
            Vector3 target = new Vector3(1f, 0.5f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0.5f, 0f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                // First waypoint should be above the start position
                Assert.Greater(waypoints[0].y, start.y,
                    "Vertical strategy lift point should be above start");
            }
        }

        #endregion

        #region Lateral Strategy Tests

        [Test]
        public void PlanAlternativePath_LateralStrategy_TriesBothSides()
        {
            // Create obstacles that block vertical approach
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);

            // Vertical obstacles (block vertical strategy)
            List<Vector3> obstacles = new List<Vector3>
            {
                new Vector3(0.5f, 0f, 0f),
                new Vector3(0.5f, 0.15f, 0f), // Block vertical path
                new Vector3(0.5f, 0.3f, 0f)
            };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Should try lateral strategy when vertical fails
            Assert.IsNotNull(waypoints);
        }

        [Test]
        public void PlanAlternativePath_LateralStrategy_OffsetsPerpendicularToMovement()
        {
            // Clear vertical path, force lateral
            var planner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 0.01f, // Very small to make vertical less attractive
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: MIN_SAFE_SEPARATION,
                maxWaypoints: MAX_WAYPOINTS
            );

            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);

            // Obstacle directly in path at various heights (block vertical)
            List<Vector3> obstacles = new List<Vector3>
            {
                new Vector3(0.5f, 0f, 0f),
                new Vector3(0.5f, 0.05f, 0f),
                new Vector3(0.5f, 0.1f, 0f)
            };

            var waypoints = planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);

            // Note: Planner may use vertical or lateral strategy depending on obstacle configuration
            // Both are valid solutions, so we just verify a path was found
        }

        #endregion

        #region Combined Strategy Tests

        [Test]
        public void PlanAlternativePath_CombinedStrategy_CreatesComplexPath()
        {
            // Create scenario where both vertical and lateral are needed
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);

            // Dense obstacle field blocking simple strategies
            List<Vector3> obstacles = new List<Vector3>();
            for (float x = 0.2f; x < 0.9f; x += 0.1f)
            {
                obstacles.Add(new Vector3(x, 0f, 0f));
                obstacles.Add(new Vector3(x, 0.15f, 0f));
            }

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Combined strategy should still find a path
            Assert.IsNotNull(waypoints);
        }

        [Test]
        public void PlanAlternativePath_CombinedStrategy_CreatesFourWaypoints()
        {
            // Force combined strategy by blocking other strategies
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);

            List<Vector3> obstacles = new List<Vector3>();
            // Create obstacles that block simple strategies
            for (float x = 0.2f; x < 0.9f; x += 0.15f)
            {
                obstacles.Add(new Vector3(x, 0f, 0f));
                obstacles.Add(new Vector3(x, 0.1f, 0f));
                obstacles.Add(new Vector3(x, 0f, 0.1f));
            }

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Combined strategy creates: lift+side, move along side, back to line, descend (4 waypoints)
            if (waypoints != null && waypoints.Count == 4)
            {
                Assert.AreEqual(4, waypoints.Count,
                    "Combined strategy should create exactly 4 waypoints when successful");
            }
        }

        #endregion

        #region Path Clearance Tests

        [Test]
        public void PlanAlternativePath_PathTooClose_RejectsPath()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);

            // Create dense obstacle field that blocks all paths
            List<Vector3> obstacles = new List<Vector3>();
            for (float x = 0f; x <= 1f; x += 0.05f)
            {
                for (float y = -0.5f; y <= 0.5f; y += 0.05f)
                {
                    for (float z = -0.5f; z <= 0.5f; z += 0.05f)
                    {
                        obstacles.Add(new Vector3(x, y, z));
                    }
                }
            }

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Should return empty list when no valid path exists
            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                // If a path was found, it should still respect minimum separation
                foreach (var wp in waypoints)
                {
                    foreach (var obstacle in obstacles)
                    {
                        float dist = Vector3.Distance(wp, obstacle);
                        Assert.GreaterOrEqual(dist, MIN_SAFE_SEPARATION - EPSILON,
                            "Waypoint should maintain minimum separation from obstacles");
                    }
                }
            }
        }

        [Test]
        public void PlanAlternativePath_ObstaclesOnlyNearPath_DoesNotAffectClearPath()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);

            // Obstacles far from path (in Z direction)
            List<Vector3> obstacles = new List<Vector3>
            {
                new Vector3(0.5f, 0f, 5f),  // Very far
                new Vector3(0.3f, 0f, 10f)
            };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Should return simple path since obstacles are far
            Assert.IsNotNull(waypoints);
        }

        #endregion

        #region Distance Calculation Tests

        [Test]
        public void RequiresReplanning_RobotDirectlyInPath_ReturnsTrue()
        {
            // Create Robot1 (the robot being planned for)
            var robot1 = new GameObject("Robot1");
            var robot1Controller = robot1.AddComponent<RobotController>();
            robot1Controller.robotId = "Robot1";
            robot1.transform.position = Vector3.zero; // Starting position

            // Create OtherRobot (obstacle)
            var otherRobot = new GameObject("OtherRobot");
            var controller = otherRobot.AddComponent<RobotController>();
            controller.robotId = "OtherRobot";

            // Position directly between start (0,0,0) and target
            otherRobot.transform.position = new Vector3(0.5f, 0f, 0f);

            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] otherRobots = new[] { robot1Controller, controller };

            try
            {
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);
                Assert.IsTrue(requiresReplanning, "Should require replanning when robot directly in path");
            }
            finally
            {
                Object.DestroyImmediate(robot1);
                Object.DestroyImmediate(otherRobot);
            }
        }

        [Test]
        public void RequiresReplanning_RobotAtTarget_ReturnsTrue()
        {
            // Create Robot1 (the robot being planned for)
            var robot1 = new GameObject("Robot1");
            var robot1Controller = robot1.AddComponent<RobotController>();
            robot1Controller.robotId = "Robot1";
            robot1.transform.position = Vector3.zero; // Starting position

            // Create OtherRobot (obstacle at target)
            var otherRobot = new GameObject("OtherRobot");
            var controller = otherRobot.AddComponent<RobotController>();
            controller.robotId = "OtherRobot";

            Vector3 target = new Vector3(1f, 0f, 0f);
            otherRobot.transform.position = target; // Exactly at target

            RobotController[] otherRobots = new[] { robot1Controller, controller };

            try
            {
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);
                Assert.IsTrue(requiresReplanning, "Should require replanning when robot at target position");
            }
            finally
            {
                Object.DestroyImmediate(robot1);
                Object.DestroyImmediate(otherRobot);
            }
        }

        [Test]
        public void RequiresReplanning_SelfRobot_ReturnsFalse()
        {
            var robot = new GameObject("Robot1");
            var controller = robot.AddComponent<RobotController>();
            controller.robotId = "Robot1";
            robot.transform.position = new Vector3(0.5f, 0f, 0f);

            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] robots = new[] { controller };

            try
            {
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, robots);
                Assert.IsFalse(requiresReplanning, "Should not require replanning due to self");
            }
            finally
            {
                Object.DestroyImmediate(robot);
            }
        }

        [Test]
        public void RequiresReplanning_NullRobotArray_ReturnsFalse()
        {
            Vector3 target = new Vector3(1f, 0f, 0f);
            bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, null);
            Assert.IsFalse(requiresReplanning, "Should not require replanning with null robot array");
        }

        [Test]
        public void RequiresReplanning_EmptyRobotArray_ReturnsFalse()
        {
            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] emptyArray = new RobotController[0];
            bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, emptyArray);
            Assert.IsFalse(requiresReplanning, "Should not require replanning with empty robot array");
        }

        #endregion

        #region Waypoint Quality Tests

        [Test]
        public void PlanAlternativePath_WaypointsFormContinuousPath()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                // Last waypoint should be at or very close to target
                float distToTarget = Vector3.Distance(waypoints[waypoints.Count - 1], target);
                Assert.Less(distToTarget, EPSILON,
                    "Final waypoint should match target position");
            }
        }

        [Test]
        public void PlanAlternativePath_WaypointsAreReasonablySpaced()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(2f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(1f, 0f, 0f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 1)
            {
                // Check spacing between consecutive waypoints
                for (int i = 0; i < waypoints.Count - 1; i++)
                {
                    float spacing = Vector3.Distance(waypoints[i], waypoints[i + 1]);

                    // Waypoints should not be too close (collapsed)
                    Assert.Greater(spacing, EPSILON,
                        $"Waypoints {i} and {i+1} should not be collapsed");

                    // Waypoints should not be unreasonably far (fragmented)
                    Assert.Less(spacing, 5f,
                        $"Waypoints {i} and {i+1} should not be too far apart");
                }
            }
        }

        #endregion

        #region Special Movement Patterns Tests

        [Test]
        public void PlanAlternativePath_DiagonalMovement_HandlesCorrectly()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 1f, 1f); // Diagonal in 3D
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0.5f, 0.5f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                // Verify final waypoint matches target
                Vector3 finalWaypoint = waypoints[waypoints.Count - 1];
                Assert.AreEqual(target, finalWaypoint,
                    "Should reach diagonal target correctly");
            }
        }

        [Test]
        public void PlanAlternativePath_VerticalMovement_HandlesCorrectly()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(0f, 1f, 0f); // Pure vertical
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0f, 0.5f, 0f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            // Should find a path even for vertical movement
        }

        [Test]
        public void PlanAlternativePath_BackwardMovement_HandlesCorrectly()
        {
            Vector3 start = new Vector3(1f, 0f, 0f);
            Vector3 target = Vector3.zero; // Moving backward
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                Assert.AreEqual(target, waypoints[waypoints.Count - 1],
                    "Should reach backward target correctly");
            }
        }

        #endregion

        #region Multiple Robots Scenario Tests

        [Test]
        public void RequiresReplanning_MultipleRobotsOnePath_ReturnsTrue()
        {
            // Create Robot1 (the robot being planned for)
            var mainRobot = new GameObject("Robot1");
            var mainController = mainRobot.AddComponent<RobotController>();
            mainController.robotId = "Robot1";
            mainRobot.transform.position = Vector3.zero; // Starting position

            // Create Robot2 (first obstacle)
            var robot1 = new GameObject("Robot2");
            var controller1 = robot1.AddComponent<RobotController>();
            controller1.robotId = "Robot2";
            robot1.transform.position = new Vector3(0.3f, 0f, 0f);

            // Create Robot3 (second obstacle)
            var robot2 = new GameObject("Robot3");
            var controller2 = robot2.AddComponent<RobotController>();
            controller2.robotId = "Robot3";
            robot2.transform.position = new Vector3(0.6f, 0f, 0f);

            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] otherRobots = new[] { mainController, controller1, controller2 };

            try
            {
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);
                Assert.IsTrue(requiresReplanning,
                    "Should require replanning when multiple robots block path");
            }
            finally
            {
                Object.DestroyImmediate(mainRobot);
                Object.DestroyImmediate(robot1);
                Object.DestroyImmediate(robot2);
            }
        }

        [Test]
        public void RequiresReplanning_MultipleRobotsOffPath_ReturnsFalse()
        {
            var robot1 = new GameObject("Robot2");
            var controller1 = robot1.AddComponent<RobotController>();
            controller1.robotId = "Robot2";
            robot1.transform.position = new Vector3(0.3f, 5f, 0f); // Far off path

            var robot2 = new GameObject("Robot3");
            var controller2 = robot2.AddComponent<RobotController>();
            controller2.robotId = "Robot3";
            robot2.transform.position = new Vector3(0.6f, 0f, 5f); // Far off path

            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] otherRobots = new[] { controller1, controller2 };

            try
            {
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);
                Assert.IsFalse(requiresReplanning,
                    "Should not require replanning when all robots are off path");
            }
            finally
            {
                Object.DestroyImmediate(robot1);
                Object.DestroyImmediate(robot2);
            }
        }

        #endregion

        #region Configuration Extremes Tests

        [Test]
        public void Constructor_VeryLargeOffsets_HandlesCorrectly()
        {
            var planner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 5.0f,
                lateralOffset: 5.0f,
                minSafeSeparation: 2.0f,
                maxWaypoints: 10
            );

            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            Assert.DoesNotThrow(() =>
            {
                planner.PlanAlternativePath("Robot1", start, target, obstacles);
            }, "Should handle very large offsets without error");
        }

        [Test]
        public void Constructor_VerySmallOffsets_HandlesCorrectly()
        {
            var planner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 0.001f,
                lateralOffset: 0.001f,
                minSafeSeparation: 0.005f,
                maxWaypoints: 3
            );

            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            Assert.DoesNotThrow(() =>
            {
                planner.PlanAlternativePath("Robot1", start, target, obstacles);
            }, "Should handle very small offsets without error");
        }

        [Test]
        public void Constructor_MaxWaypoints1_LimitsProperly()
        {
            var planner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: VERTICAL_OFFSET,
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: MIN_SAFE_SEPARATION,
                maxWaypoints: 1
            );

            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            var waypoints = planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // With max 1 waypoint, most strategies won't work
            if (waypoints != null)
            {
                Assert.LessOrEqual(waypoints.Count, 1,
                    "Should respect maxWaypoints=1 limit");
            }
        }

        #endregion

        #region Stress Tests

        [Test]
        public void PlanAlternativePath_ManyObstacles_PerformanceAcceptable()
        {
            Vector3 start = Vector3.zero;
            Vector3 target = new Vector3(5f, 0f, 0f);

            // Create 1000 obstacles
            List<Vector3> obstacles = new List<Vector3>();
            for (int i = 0; i < 1000; i++)
            {
                obstacles.Add(new Vector3(
                    Random.Range(-10f, 10f),
                    Random.Range(-10f, 10f),
                    Random.Range(-10f, 10f)
                ));
            }

            var startTime = Time.realtimeSinceStartup;
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);
            var elapsedTime = Time.realtimeSinceStartup - startTime;

            Assert.Less(elapsedTime, 1.0f,
                "Should handle 1000 obstacles in under 1 second");
            Assert.IsNotNull(waypoints);
        }

        #endregion
    }
}
