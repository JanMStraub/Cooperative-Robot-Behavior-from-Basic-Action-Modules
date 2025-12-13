using NUnit.Framework;
using Robotics;
using UnityEngine;

namespace Tests.EditMode
{
    /// <summary>
    /// Tests for GraspOptions data structure and presets
    /// </summary>
    public class GraspOptionsTests
    {
        [Test]
        public void Default_Preset_Has_Correct_Values()
        {
            // Arrange & Act
            var options = GraspOptions.Default;

            // Assert
            Assert.IsTrue(options.useGraspPlanning, "Default should use grasp planning");
            Assert.IsTrue(options.openGripperOnSet, "Default should open gripper on set");
            Assert.IsTrue(options.closeGripperOnReach, "Default should close gripper on reach");
            Assert.IsNull(options.approach, "Default should have null approach (auto-determine)");
        }

        [Test]
        public void MoveOnly_Preset_Has_Correct_Values()
        {
            // Arrange & Act
            var options = GraspOptions.MoveOnly;

            // Assert
            Assert.IsFalse(options.useGraspPlanning, "MoveOnly should not use grasp planning");
            Assert.IsFalse(options.openGripperOnSet, "MoveOnly should not open gripper on set");
            Assert.IsFalse(options.closeGripperOnReach, "MoveOnly should not close gripper on reach");
            Assert.IsNull(options.approach, "MoveOnly should have null approach");
        }

        [Test]
        public void Custom_Options_Are_Respected()
        {
            // Arrange & Act
            var customOptions = new GraspOptions
            {
                useGraspPlanning = true,
                openGripperOnSet = false,
                closeGripperOnReach = true,
                approach = GraspApproach.Top
            };

            // Assert
            Assert.IsTrue(customOptions.useGraspPlanning);
            Assert.IsFalse(customOptions.openGripperOnSet);
            Assert.IsTrue(customOptions.closeGripperOnReach);
            Assert.AreEqual(GraspApproach.Top, customOptions.approach);
        }

        [Test]
        public void Default_And_MoveOnly_Are_Different()
        {
            // Arrange
            var defaultOptions = GraspOptions.Default;
            var moveOnlyOptions = GraspOptions.MoveOnly;

            // Assert
            Assert.AreNotEqual(defaultOptions.useGraspPlanning, moveOnlyOptions.useGraspPlanning);
            Assert.AreNotEqual(defaultOptions.openGripperOnSet, moveOnlyOptions.openGripperOnSet);
            Assert.AreNotEqual(defaultOptions.closeGripperOnReach, moveOnlyOptions.closeGripperOnReach);
        }

        [Test]
        public void Can_Override_Approach_In_Default()
        {
            // Arrange
            var options = GraspOptions.Default;

            // Act
            options.approach = GraspApproach.Side;

            // Assert
            Assert.AreEqual(GraspApproach.Side, options.approach);
            Assert.IsTrue(options.useGraspPlanning, "Other properties should remain unchanged");
        }
    }
}
