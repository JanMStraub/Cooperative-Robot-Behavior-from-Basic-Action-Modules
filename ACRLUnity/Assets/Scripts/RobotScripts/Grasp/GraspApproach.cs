namespace Robotics.Grasp
{
    /// <summary>
    /// Defines the approach direction for grasp planning.
    /// Used to generate diverse grasp candidates from different angles.
    /// </summary>
    public enum GraspApproach
    {
        /// <summary>
        /// Approach from above the object (gravity-aligned, most stable)
        /// </summary>
        Top = 0,

        /// <summary>
        /// Approach from the front of the object (good for horizontal surfaces)
        /// </summary>
        Front = 1,

        /// <summary>
        /// Approach from the side of the object (useful for tight spaces)
        /// </summary>
        Side = 2,
    }
}
