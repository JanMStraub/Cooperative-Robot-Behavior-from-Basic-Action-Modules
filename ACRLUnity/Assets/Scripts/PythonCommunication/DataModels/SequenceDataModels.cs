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
        public bool success;

        public uint request_id;

        public string sequence_id;

        public int total_commands;

        public int completed_commands;

        public List<CommandResult> results;

        public List<ParsedCommand> parsed_commands;

        public string original_command;

        public float total_duration_ms;

        public string error;
    }

    /// <summary>
    /// Result of a single command execution.
    /// </summary>
    [Serializable]
    public class CommandResult
    {
        public int index;

        public string operation;

        public bool success;

        public object result;

        public string error;

        public float duration_ms;
    }

    /// <summary>
    /// A parsed command from natural language.
    /// </summary>
    [Serializable]
    public class ParsedCommand
    {
        public string operation;

        public Dictionary<string, object> @params;
    }
}
