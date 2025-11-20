using System;
using System.Collections.Generic;

namespace PythonCommunication
{
    /// <summary>
    /// Result of a sequence execution from Python.
    /// </summary>
    [Serializable]
    public class SequenceResult
    {
        /// <summary>
        /// Whether the sequence executed successfully
        /// </summary>
        public bool success;

        /// <summary>
        /// Unique identifier for this sequence execution
        /// </summary>
        public string sequence_id;

        /// <summary>
        /// Total number of commands in the sequence
        /// </summary>
        public int total_commands;

        /// <summary>
        /// Number of commands that completed successfully
        /// </summary>
        public int completed_commands;

        /// <summary>
        /// Results for each command in the sequence
        /// </summary>
        public List<CommandResult> results;

        /// <summary>
        /// Original parsed commands
        /// </summary>
        public List<ParsedCommand> parsed_commands;

        /// <summary>
        /// Original natural language command
        /// </summary>
        public string original_command;

        /// <summary>
        /// Total execution time in milliseconds
        /// </summary>
        public float total_duration_ms;

        /// <summary>
        /// Error message if sequence failed
        /// </summary>
        public string error;
    }

    /// <summary>
    /// Result of a single command execution.
    /// </summary>
    [Serializable]
    public class CommandResult
    {
        /// <summary>
        /// Index of this command in the sequence
        /// </summary>
        public int index;

        /// <summary>
        /// Operation name that was executed
        /// </summary>
        public string operation;

        /// <summary>
        /// Whether this command succeeded
        /// </summary>
        public bool success;

        /// <summary>
        /// Result data from the operation
        /// </summary>
        public object result;

        /// <summary>
        /// Error message if command failed
        /// </summary>
        public string error;

        /// <summary>
        /// Execution time in milliseconds
        /// </summary>
        public float duration_ms;
    }

    /// <summary>
    /// A parsed command from natural language.
    /// </summary>
    [Serializable]
    public class ParsedCommand
    {
        /// <summary>
        /// Operation name (e.g., "move_to_coordinate")
        /// </summary>
        public string operation;

        /// <summary>
        /// Command parameters
        /// </summary>
        public Dictionary<string, object> @params;
    }
}
