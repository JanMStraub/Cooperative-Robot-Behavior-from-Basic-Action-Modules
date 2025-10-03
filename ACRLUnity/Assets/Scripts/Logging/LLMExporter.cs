using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Logging
{
    /// <summary>
    /// Lightweight LLM training data exporter
    /// Exports logs in JSONL format (standard for LLM training)
    /// Replaces: LLMDataExporter with 5 formats (now just JSONL + conversational)
    /// </summary>
    public class LLMExporter
    {
        /// <summary>
        /// Export logs to JSONL format for LLM training
        /// </summary>
        public static void ExportToJSONL(string sourceLogFile, string outputFile)
        {
            if (!File.Exists(sourceLogFile))
            {
                Debug.LogError($"Source log file not found: {sourceLogFile}");
                return;
            }

            try
            {
                var lines = File.ReadAllLines(sourceLogFile);
                var entries = new List<LogEntry>();

                // Parse all log entries
                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    try
                    {
                        var entry = JsonUtility.FromJson<LogEntry>(line);
                        if (entry != null)
                            entries.Add(entry);
                    }
                    catch
                    {
                        // Skip malformed entries
                    }
                }

                // Write clean JSONL output
                using (var writer = new StreamWriter(outputFile))
                {
                    foreach (var entry in entries)
                    {
                        string json = JsonUtility.ToJson(entry);
                        writer.WriteLine(json);
                    }
                }

                Debug.Log($"Exported {entries.Count} entries to {outputFile}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Export failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Export to conversational format for chat model training
        /// Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        /// </summary>
        public static void ExportToConversational(string sourceLogFile, string outputFile)
        {
            if (!File.Exists(sourceLogFile))
            {
                Debug.LogError($"Source log file not found: {sourceLogFile}");
                return;
            }

            try
            {
                var lines = File.ReadAllLines(sourceLogFile);
                var conversations = new List<Dictionary<string, object>>();

                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    try
                    {
                        var entry = JsonUtility.FromJson<LogEntry>(line);
                        if (entry == null || entry.logType == "session")
                            continue;

                        var conversation = new Dictionary<string, object>
                        {
                            ["messages"] = new List<Dictionary<string, string>>
                            {
                                new Dictionary<string, string>
                                {
                                    ["role"] = "user",
                                    ["content"] = entry.trainingPrompt ?? "What happened?",
                                },
                                new Dictionary<string, string>
                                {
                                    ["role"] = "assistant",
                                    ["content"] = entry.trainingResponse ?? "Action completed",
                                },
                            },
                            ["metadata"] = new Dictionary<string, object>
                            {
                                ["difficulty"] = entry.difficultyLevel,
                                ["timestamp"] = entry.timestamp,
                                ["learning_points"] = entry.learningPoints,
                            },
                        };

                        conversations.Add(conversation);
                    }
                    catch
                    {
                        // Skip malformed entries
                    }
                }

                // Write as JSON array
                using (var writer = new StreamWriter(outputFile))
                {
                    writer.WriteLine("[");
                    for (int i = 0; i < conversations.Count; i++)
                    {
                        string json = JsonUtility.ToJson(conversations[i]);
                        writer.Write("  " + json);
                        if (i < conversations.Count - 1)
                            writer.WriteLine(",");
                        else
                            writer.WriteLine();
                    }
                    writer.WriteLine("]");
                }

                Debug.Log($"Exported {conversations.Count} conversations to {outputFile}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Conversational export failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Generate training summary statistics
        /// </summary>
        public static Dictionary<string, object> GenerateStatistics(string logFile)
        {
            var stats = new Dictionary<string, object>();

            try
            {
                var lines = File.ReadAllLines(logFile);
                var actions = new List<RobotAction>();
                var scenes = new List<SceneSnapshot>();

                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    var entry = JsonUtility.FromJson<LogEntry>(line);
                    if (entry == null)
                        continue;

                    if (entry.action != null)
                        actions.Add(entry.action);
                    if (entry.scene != null)
                        scenes.Add(entry.scene);
                }

                stats["total_entries"] = lines.Length;
                stats["total_actions"] = actions.Count;
                stats["total_scenes"] = scenes.Count;
                stats["success_rate"] =
                    actions.Count > 0 ? actions.Count(a => a.success) / (float)actions.Count : 0f;
                stats["action_types"] = actions
                    .GroupBy(a => a.type)
                    .ToDictionary(g => g.Key.ToString(), g => g.Count());
                stats["complexity_distribution"] = actions
                    .GroupBy(a => a.complexityLevel)
                    .ToDictionary(g => $"level_{g.Key}", g => g.Count());
                stats["average_duration"] =
                    actions.Count > 0 ? actions.Average(a => a.duration) : 0f;
                stats["unique_robots"] = actions.SelectMany(a => a.robotIds).Distinct().Count();

                Debug.Log("Statistics generated:");
                foreach (var stat in stats)
                {
                    Debug.Log($"  {stat.Key}: {stat.Value}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Statistics generation failed: {ex.Message}");
            }

            return stats;
        }

        /// <summary>
        /// Filter logs by criteria
        /// </summary>
        public static void FilterLogs(
            string sourceFile,
            string outputFile,
            ActionType? typeFilter = null,
            bool? successFilter = null,
            int? minComplexity = null
        )
        {
            try
            {
                var lines = File.ReadAllLines(sourceFile);
                int filtered = 0;

                using (var writer = new StreamWriter(outputFile))
                {
                    foreach (var line in lines)
                    {
                        if (string.IsNullOrWhiteSpace(line))
                            continue;

                        var entry = JsonUtility.FromJson<LogEntry>(line);
                        if (entry?.action == null)
                            continue;

                        // Apply filters
                        if (typeFilter.HasValue && entry.action.type != typeFilter.Value)
                            continue;
                        if (successFilter.HasValue && entry.action.success != successFilter.Value)
                            continue;
                        if (
                            minComplexity.HasValue
                            && entry.action.complexityLevel < minComplexity.Value
                        )
                            continue;

                        writer.WriteLine(line);
                        filtered++;
                    }
                }

                Debug.Log($"Filtered {filtered} entries to {outputFile}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Filtering failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Quick export from RobotLogger instance
        /// </summary>
        public static void QuickExport(string format = "jsonl")
        {
            var logger = RobotLogger.Instance;
            if (logger == null)
            {
                Debug.LogError("RobotLogger instance not found");
                return;
            }

            string logDir = Path.Combine(Application.persistentDataPath, "RobotLogs");
            if (!Directory.Exists(logDir))
            {
                Debug.LogError("Log directory not found");
                return;
            }

            // Find most recent log file
            var files = Directory
                .GetFiles(logDir, "robot_actions_*.jsonl")
                .OrderByDescending(f => File.GetLastWriteTime(f))
                .ToArray();

            if (files.Length == 0)
            {
                Debug.LogError("No log files found");
                return;
            }

            string sourceFile = files[0];
            string outputFile =
                format.ToLower() == "conversational"
                    ? sourceFile.Replace(".jsonl", "_conversational.json")
                    : sourceFile.Replace(".jsonl", "_export.jsonl");

            if (format.ToLower() == "conversational")
            {
                ExportToConversational(sourceFile, outputFile);
            }
            else
            {
                ExportToJSONL(sourceFile, outputFile);
            }

            Debug.Log($"Quick export complete: {outputFile}");
        }
    }

#if UNITY_EDITOR
    /// <summary>
    /// Unity Editor menu integration
    /// </summary>
    public static class ExporterMenu
    {
        [UnityEditor.MenuItem("Tools/Robot Logging/Export to JSONL")]
        private static void ExportJSONL()
        {
            LLMExporter.QuickExport("jsonl");
        }

        [UnityEditor.MenuItem("Tools/Robot Logging/Export to Conversational")]
        private static void ExportConversational()
        {
            LLMExporter.QuickExport("conversational");
        }

        [UnityEditor.MenuItem("Tools/Robot Logging/Generate Statistics")]
        private static void GenerateStats()
        {
            string logDir = Path.Combine(Application.persistentDataPath, "RobotLogs");
            var files = Directory
                .GetFiles(logDir, "robot_actions_*.jsonl")
                .OrderByDescending(f => File.GetLastWriteTime(f))
                .ToArray();

            if (files.Length > 0)
            {
                LLMExporter.GenerateStatistics(files[0]);
            }
        }

        [UnityEditor.MenuItem("Tools/Robot Logging/Open Log Directory")]
        private static void OpenLogDirectory()
        {
            string logDir = Path.Combine(Application.persistentDataPath, "RobotLogs");
            if (Directory.Exists(logDir))
            {
                UnityEditor.EditorUtility.RevealInFinder(logDir);
            }
        }
    }
#endif
}
