using System.Collections.Generic;
using UnityEngine;

namespace PythonCommunication
{
    /// <summary>
    /// Result of a coordination verification check.
    /// </summary>
    public struct VerificationResult
    {
        public bool isSafe;
        public string reason;
        public List<string> warnings;

        public VerificationResult(bool isSafe, string reason = "", List<string> warnings = null)
        {
            this.isSafe = isSafe;
            this.reason = reason ?? "";
            this.warnings = warnings ?? new List<string>();
        }
    }

    /// <summary>
    /// Interface for coordination verification strategies.
    /// Verifies whether a robot movement is safe in the current coordination context.
    /// </summary>
    public interface ICoordinationVerifier
    {
        /// <summary>
        /// Verify if a robot movement is safe.
        /// </summary>
        /// <param name="robotId">Robot identifier</param>
        /// <param name="targetPosition">Target position for movement</param>
        /// <param name="currentPosition">Current robot position</param>
        /// <returns>Verification result with safety status and reason</returns>
        VerificationResult VerifyMovement(
            string robotId,
            Vector3 targetPosition,
            Vector3 currentPosition
        );

        /// <summary>
        /// Check if the verifier is currently available and operational.
        /// </summary>
        bool IsAvailable { get; }

        /// <summary>
        /// Get the name of this verifier for logging.
        /// </summary>
        string VerifierName { get; }
    }
}
