using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Simulation.CoordinationStrategies;
using Robotics;

namespace Tests.PlayMode
{
    /// <summary>
    /// Tests for WaypointCollisionAvoidancePlanner (Phase 3).
    /// Validates vertical offset, lateral offset, and combined planning strategies.
    /// </summary>
    public class CollisionAvoidancePlannerTests
    {
        private WaypointCollisionAvoidancePlanner _planner;
        private const float VERTICAL_OFFSET = 0.15f;
        private const float LATERAL_OFFSET = 0.1f;
        private const float MIN_SAFE_SEPARATION = 0.2f;
        private const int MAX_WAYPOINTS = 5;

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

        #region Constructor Tests

        [Test]
        public void Constructor_DefaultParameters_InitializesCorrectly()
        {
            var defaultPlanner = new WaypointCollisionAvoidancePlanner();
            Assert.IsNotNull(defaultPlanner);
        }

        [Test]
        public void Constructor_CustomParameters_InitializesCorrectly()
        {
            var customPlanner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 0.25f,
                lateralOffset: 0.15f,
                minSafeSeparation: 0.3f,
                maxWaypoints: 10
            );
            Assert.IsNotNull(customPlanner);
        }

        [Test]
        public void Constructor_NegativeValues_ClampsToMinimum()
        {
            // Should not throw, should clamp to minimum values
            Assert.DoesNotThrow(() =>
            {
                var planner = new WaypointCollisionAvoidancePlanner(
                    verticalOffset: -0.1f,
                    lateralOffset: -0.1f,
                    minSafeSeparation: -0.1f,
                    maxWaypoints: -5
                );
            });
        }

        #endregion

        #region Vertical Offset Planning Tests

        [Test]
        public void PlanAlternativePath_VerticalStrategy_CreatesUpThenDownPath()
        {
            // Arrange
            string robotId = "Robot1";
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            // Act
            var waypoints = _planner.PlanAlternativePath(robotId, start, target, obstacles);

            // Assert
            Assert.IsNotNull(waypoints);
            Assert.IsTrue(waypoints.Count >= 2, "Should have at least 2 waypoints (up and down)");

            // First waypoint should be above start
            Assert.Greater(waypoints[0].y, start.y, "First waypoint should move upward");

            // Last waypoint should be at target height or close to it
            Assert.LessOrEqual(
                Mathf.Abs(waypoints[waypoints.Count - 1].y - target.y),
                0.01f,
                "Final waypoint should return to target height"
            );
        }

        [Test]
        public void PlanAlternativePath_NoObstacles_ReturnsDirectPath()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3>();

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert - With no obstacles, might return null or minimal waypoints
            if (waypoints != null)
            {
                Assert.LessOrEqual(waypoints.Count, 2, "Should have minimal waypoints with no obstacles");
            }
        }

        [Test]
        public void PlanAlternativePath_MultipleObstacles_AvoidsAll()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3>
            {
                new Vector3(0.3f, 0f, 0f),
                new Vector3(0.6f, 0f, 0f),
                new Vector3(0.9f, 0f, 0f)
            };

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert
            Assert.IsNotNull(waypoints);

            // Verify waypoints avoid obstacles
            foreach (var waypoint in waypoints)
            {
                foreach (var obstacle in obstacles)
                {
                    float distance = Vector3.Distance(waypoint, obstacle);
                    Assert.GreaterOrEqual(
                        distance,
                        MIN_SAFE_SEPARATION - 0.01f, // Small tolerance
                        $"Waypoint {waypoint} too close to obstacle {obstacle}"
                    );
                }
            }
        }

        #endregion

        #region Lateral Offset Planning Tests

        [Test]
        public void RequiresReplanning_ObstacleInPath_ReturnsTrue()
        {
            // Arrange
            // Create Robot1 (the robot we're checking replanning for)
            var robot1 = new GameObject("Robot1");
            var robot1Controller = robot1.AddComponent<RobotController>();
            robot1Controller.robotId = "Robot1";
            robot1.transform.position = new Vector3(0f, 0f, 0f); // Start position

            // Set up end effector
            var endEffector1 = new GameObject("EndEffector1");
            endEffector1.transform.SetParent(robot1.transform);
            endEffector1.transform.localPosition = Vector3.zero;
            robot1Controller.endEffectorBase = endEffector1.transform;

            // Create obstacle robot
            var otherRobot = new GameObject("OtherRobot");
            var otherController = otherRobot.AddComponent<RobotController>();
            otherController.robotId = "OtherRobot";
            otherRobot.transform.position = new Vector3(0.5f, 0f, 0f); // Directly in path

            // Set up end effector for other robot
            var endEffector2 = new GameObject("EndEffector2");
            endEffector2.transform.SetParent(otherRobot.transform);
            endEffector2.transform.localPosition = Vector3.zero;
            otherController.endEffectorBase = endEffector2.transform;

            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] otherRobots = new[] { robot1Controller, otherController };

            try
            {
                // Act
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);

                // Assert
                Assert.IsTrue(requiresReplanning, "Should require replanning when robot blocks path");
            }
            finally
            {
                Object.DestroyImmediate(robot1);
                Object.DestroyImmediate(otherRobot);
            }
        }

        [Test]
        public void RequiresReplanning_NoObstacles_ReturnsFalse()
        {
            // Arrange
            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] otherRobots = new RobotController[0];

            // Act
            bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);

            // Assert
            Assert.IsFalse(requiresReplanning, "Should not require replanning with no obstacles");
        }

        [Test]
        public void RequiresReplanning_ObstacleFarFromPath_ReturnsFalse()
        {
            // Arrange
            // Create Robot1
            var robot1 = new GameObject("Robot1");
            var robot1Controller = robot1.AddComponent<RobotController>();
            robot1Controller.robotId = "Robot1";
            robot1.transform.position = new Vector3(0f, 0f, 0f); // Start position

            // Set up end effector
            var endEffector1 = new GameObject("EndEffector1");
            endEffector1.transform.SetParent(robot1.transform);
            endEffector1.transform.localPosition = Vector3.zero;
            robot1Controller.endEffectorBase = endEffector1.transform;

            var otherRobot = new GameObject("OtherRobot");
            var controller = otherRobot.AddComponent<RobotController>();
            controller.robotId = "OtherRobot";
            otherRobot.transform.position = new Vector3(0.5f, 5f, 0f); // Far above

            // Set up end effector for other robot
            var endEffector2 = new GameObject("EndEffector2");
            endEffector2.transform.SetParent(otherRobot.transform);
            endEffector2.transform.localPosition = Vector3.zero;
            controller.endEffectorBase = endEffector2.transform;

            Vector3 target = new Vector3(1f, 0f, 0f);
            RobotController[] otherRobots = new[] { robot1Controller, controller };

            try
            {
                // Act
                bool requiresReplanning = _planner.RequiresReplanning("Robot1", target, otherRobots);

                // Assert
                Assert.IsFalse(requiresReplanning, "Should not require replanning when obstacle is far from path");
            }
            finally
            {
                Object.DestroyImmediate(robot1);
                Object.DestroyImmediate(otherRobot);
            }
        }

        #endregion

        #region Path Validation Tests

        [Test]
        public void PlanAlternativePath_StartEqualsTarget_ReturnsNull()
        {
            // Arrange
            Vector3 position = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3>();

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", position, position, obstacles);

            // Assert - Should return null or empty when start equals target
            Assert.IsTrue(waypoints == null || waypoints.Count == 0,
                "Should not create waypoints when start equals target");
        }

        [Test]
        public void PlanAlternativePath_VeryShortDistance_HandlesCorrectly()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(0.01f, 0f, 0f); // 1cm away
            List<Vector3> obstacles = new List<Vector3>();

            // Act & Assert - Should not crash
            Assert.DoesNotThrow(() =>
            {
                var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);
            });
        }

        [Test]
        public void PlanAlternativePath_VeryLongDistance_HandlesCorrectly()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(100f, 0f, 0f); // Very far
            List<Vector3> obstacles = new List<Vector3> { new Vector3(50f, 0f, 0f) };

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert
            Assert.IsNotNull(waypoints);
            Assert.LessOrEqual(waypoints.Count, MAX_WAYPOINTS,
                "Should respect maximum waypoints limit");
        }

        #endregion

        #region Configuration Tests

        [Test]
        public void MaxWaypoints_LimitEnforced()
        {
            // Arrange - Create planner with max 3 waypoints
            var limitedPlanner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: VERTICAL_OFFSET,
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: MIN_SAFE_SEPARATION,
                maxWaypoints: 3
            );

            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(10f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3>
            {
                new Vector3(2f, 0f, 0f),
                new Vector3(4f, 0f, 0f),
                new Vector3(6f, 0f, 0f),
                new Vector3(8f, 0f, 0f)
            };

            // Act
            var waypoints = limitedPlanner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert
            if (waypoints != null)
            {
                Assert.LessOrEqual(waypoints.Count, 3,
                    "Should respect configured maximum waypoints");
            }
        }

        [Test]
        public void VerticalOffset_AffectsWaypointHeight()
        {
            // Arrange - Create planners with different vertical offsets
            var smallOffsetPlanner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 0.05f,
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: MIN_SAFE_SEPARATION,
                maxWaypoints: MAX_WAYPOINTS
            );

            var largeOffsetPlanner = new WaypointCollisionAvoidancePlanner(
                verticalOffset: 0.5f,
                lateralOffset: LATERAL_OFFSET,
                minSafeSeparation: MIN_SAFE_SEPARATION,
                maxWaypoints: MAX_WAYPOINTS
            );

            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            // Act
            var smallOffsetWaypoints = smallOffsetPlanner.PlanAlternativePath("Robot1", start, target, obstacles);
            var largeOffsetWaypoints = largeOffsetPlanner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert - Large offset should create higher waypoints
            if (smallOffsetWaypoints != null && largeOffsetWaypoints != null &&
                smallOffsetWaypoints.Count > 0 && largeOffsetWaypoints.Count > 0)
            {
                float smallMaxHeight = 0f;
                float largeMaxHeight = 0f;

                foreach (var wp in smallOffsetWaypoints)
                    smallMaxHeight = Mathf.Max(smallMaxHeight, wp.y);

                foreach (var wp in largeOffsetWaypoints)
                    largeMaxHeight = Mathf.Max(largeMaxHeight, wp.y);

                Assert.Greater(largeMaxHeight, smallMaxHeight,
                    "Larger vertical offset should create higher waypoints");
            }
        }

        #endregion

        #region Edge Cases

        [Test]
        public void PlanAlternativePath_NullObstacles_HandlesGracefully()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);

            // Act & Assert - Should not crash with null obstacles
            Assert.DoesNotThrow(() =>
            {
                var waypoints = _planner.PlanAlternativePath("Robot1", start, target, null);
            });
        }

        [Test]
        public void PlanAlternativePath_EmptyRobotId_HandlesGracefully()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3>();

            // Act & Assert - Should not crash with empty robot ID
            Assert.DoesNotThrow(() =>
            {
                var waypoints = _planner.PlanAlternativePath("", start, target, obstacles);
            });
        }

        [Test]
        public void PlanAlternativePath_ObstacleAtStartPosition_HandlesCorrectly()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { start }; // Obstacle at start

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert - Should create a path that moves away from the obstacle
            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                Assert.AreNotEqual(start, waypoints[0],
                    "First waypoint should move away from obstacle at start");
            }
        }

        [Test]
        public void PlanAlternativePath_ObstacleAtTargetPosition_HandlesCorrectly()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0f, 0f);
            List<Vector3> obstacles = new List<Vector3> { target }; // Obstacle at target

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert - Should attempt to create a path, even if final approach is blocked
            // The planner should try to get as close as possible
            Assert.IsNotNull(waypoints);
        }

        #endregion

        #region Multiple Robot Scenarios

        [Test]
        public void PlanAlternativePath_TwoRobotsWithDifferentIds_CreatesIndependentPaths()
        {
            // Arrange
            Vector3 start1 = new Vector3(0f, 0f, 0f);
            Vector3 target1 = new Vector3(1f, 0f, 0f);
            Vector3 start2 = new Vector3(0f, 0f, 1f);
            Vector3 target2 = new Vector3(1f, 0f, 1f);
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0f, 0f) };

            // Act
            var waypoints1 = _planner.PlanAlternativePath("Robot1", start1, target1, obstacles);
            var waypoints2 = _planner.PlanAlternativePath("Robot2", start2, target2, new List<Vector3>());

            // Assert - Both should succeed independently
            Assert.IsNotNull(waypoints1);
            Assert.IsNotNull(waypoints2);
        }

        #endregion

        #region 3D Space Tests

        [Test]
        public void PlanAlternativePath_3DMovement_HandlesAllAxes()
        {
            // Arrange
            Vector3 start = new Vector3(0f, 0f, 0f);
            Vector3 target = new Vector3(1f, 0.5f, 1f); // Movement in X, Y, and Z
            List<Vector3> obstacles = new List<Vector3> { new Vector3(0.5f, 0.25f, 0.5f) };

            // Act
            var waypoints = _planner.PlanAlternativePath("Robot1", start, target, obstacles);

            // Assert
            Assert.IsNotNull(waypoints);
            if (waypoints.Count > 0)
            {
                // Verify waypoints are in 3D space
                foreach (var waypoint in waypoints)
                {
                    Assert.IsTrue(
                        !float.IsNaN(waypoint.x) && !float.IsNaN(waypoint.y) && !float.IsNaN(waypoint.z),
                        "Waypoint should have valid 3D coordinates"
                    );
                }
            }
        }

        #endregion
    }
}
