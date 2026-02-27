using System.Collections.Generic;
using PythonCommunication;
using PythonCommunication.DataModels;
using UnityEditor;
using UnityEngine;

namespace ACRLEditor
{
    /// <summary>
    /// Custom inspector for AutoRTManager.
    /// Provides UI for task generation control, loop management, and task approval.
    /// </summary>
    [CustomEditor(typeof(AutoRTManager))]
    public class AutoRTManagerEditor : Editor
    {
        private AutoRTManager _manager;
        private Vector2 _taskScrollPosition;
        private GUIStyle _taskCardStyle;
        private GUIStyle _headerStyle;
        private GUIStyle _buttonStyle;
        private bool _stylesInitialized = false;

        private void OnEnable()
        {
            _manager = (AutoRTManager)target;
        }

        public override void OnInspectorGUI()
        {
            // Initialize styles
            if (!_stylesInitialized)
            {
                InitializeStyles();
                _stylesInitialized = true;
            }

            // Draw default inspector
            DrawDefaultInspector();

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("AutoRT Controls", _headerStyle);
            EditorGUILayout.Space(5);

            // Connection status
            DrawConnectionStatus();

            EditorGUILayout.Space(10);

            // Control buttons (only enabled when connected and in play mode)
            bool canControl = _manager != null && _manager.IsConnected && Application.isPlaying;

            EditorGUI.BeginDisabledGroup(!canControl);

            // Manual generation button
            if (GUILayout.Button("Generate Tasks Now", GUILayout.Height(30)))
            {
                _manager.GenerateTasks();
            }

            EditorGUILayout.Space(5);

            // Loop control buttons
            EditorGUILayout.BeginHorizontal();

            if (_manager.LoopRunning)
            {
                if (GUILayout.Button("Stop Continuous Loop", GUILayout.Height(30)))
                {
                    _manager.StopLoop();
                }
            }
            else
            {
                if (GUILayout.Button("Start Continuous Loop", GUILayout.Height(30)))
                {
                    _manager.StartLoop();
                }
            }

            EditorGUILayout.EndHorizontal();

            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space(10);

            // Loop status display
            DrawLoopStatus();

            EditorGUILayout.Space(10);

            // Pending tasks section
            DrawPendingTasks();

            // Auto-repaint in play mode
            if (Application.isPlaying && _manager.Config != null && _manager.Config.autoRefresh)
            {
                Repaint();
            }
        }

        private void InitializeStyles()
        {
            _headerStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 14,
                alignment = TextAnchor.MiddleCenter,
            };

            _taskCardStyle = new GUIStyle(EditorStyles.helpBox)
            {
                padding = new RectOffset(10, 10, 10, 10),
                margin = new RectOffset(0, 0, 5, 5),
            };

            _buttonStyle = new GUIStyle(GUI.skin.button) { fontStyle = FontStyle.Bold };
        }

        private void DrawConnectionStatus()
        {
            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.LabelField("Connection Status:", GUILayout.Width(120));

            if (_manager.IsConnected)
            {
                GUI.color = Color.green;
                EditorGUILayout.LabelField("● CONNECTED", EditorStyles.boldLabel);
            }
            else
            {
                GUI.color = Color.yellow;
                EditorGUILayout.LabelField("○ Not Connected", EditorStyles.label);
            }

            GUI.color = Color.white;

            EditorGUILayout.EndHorizontal();

            // Status message
            if (!string.IsNullOrEmpty(_manager.StatusMessage))
            {
                EditorGUILayout.LabelField(
                    "Status:",
                    _manager.StatusMessage,
                    EditorStyles.wordWrappedLabel
                );
            }
        }

        private void DrawLoopStatus()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            EditorGUILayout.LabelField("Loop Status", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("State:", GUILayout.Width(80));

            if (_manager.LoopRunning)
            {
                GUI.color = Color.green;
                EditorGUILayout.LabelField("RUNNING", EditorStyles.boldLabel);
            }
            else
            {
                GUI.color = Color.gray;
                EditorGUILayout.LabelField("STOPPED", EditorStyles.label);
            }

            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();

            // Loop configuration (if available)
            if (_manager.Config != null)
            {
                EditorGUILayout.LabelField($"Delay: {_manager.Config.loopDelaySeconds}s");
                EditorGUILayout.LabelField($"Strategy: {_manager.Config.strategy}");
                EditorGUILayout.LabelField($"Robots: {_manager.Config.GetRobotIdsString()}");
            }

            EditorGUILayout.EndVertical();
        }

        private void DrawPendingTasks()
        {
            EditorGUILayout.LabelField("Pending Tasks", _headerStyle);

            List<ProposedTask> tasks = _manager.PendingTasks;

            if (tasks == null || tasks.Count == 0)
            {
                EditorGUILayout.HelpBox(
                    "No pending tasks. Click 'Generate Tasks Now' or start the continuous loop.",
                    MessageType.Info
                );
                return;
            }

            EditorGUILayout.LabelField(
                $"Tasks awaiting approval: {tasks.Count}",
                EditorStyles.miniLabel
            );
            EditorGUILayout.Space(5);

            // Clear all button
            if (GUILayout.Button("Clear All Tasks"))
            {
                if (
                    EditorUtility.DisplayDialog(
                        "Clear All Tasks",
                        $"Are you sure you want to clear all {tasks.Count} pending tasks?",
                        "Yes",
                        "No"
                    )
                )
                {
                    _manager.ClearPendingTasks();
                }
            }

            EditorGUILayout.Space(5);

            // Scrollable task list
            _taskScrollPosition = EditorGUILayout.BeginScrollView(
                _taskScrollPosition,
                GUILayout.MaxHeight(400)
            );

            // Iterate through tasks and draw cards
            for (int i = 0; i < tasks.Count; i++)
            {
                DrawTaskCard(tasks[i], i);
            }

            EditorGUILayout.EndScrollView();
        }

        private void DrawTaskCard(ProposedTask task, int index)
        {
            if (task == null)
                return;

            EditorGUILayout.BeginVertical(_taskCardStyle);

            // Task header
            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.LabelField(
                $"Task #{index + 1}",
                EditorStyles.boldLabel,
                GUILayout.Width(80)
            );

            // Complexity badge
            GUI.color = GetComplexityColor(task.estimated_complexity);
            EditorGUILayout.LabelField(
                $"Complexity: {task.estimated_complexity}",
                EditorStyles.miniLabel,
                GUILayout.Width(100)
            );
            GUI.color = Color.white;

            // Robots required
            EditorGUILayout.LabelField(
                $"Robots: {task.RobotCount}",
                EditorStyles.miniLabel,
                GUILayout.Width(80)
            );

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(3);

            // Task description
            EditorGUILayout.LabelField("Description:", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(task.description, EditorStyles.wordWrappedLabel);

            EditorGUILayout.Space(3);

            // Operations list
            if (task.operations != null && task.operations.Count > 0)
            {
                EditorGUILayout.LabelField(
                    $"Operations ({task.operations.Count}):",
                    EditorStyles.boldLabel
                );

                foreach (var op in task.operations)
                {
                    if (op != null)
                    {
                        EditorGUILayout.LabelField(
                            $"  • {op.type} (Robot: {op.robot_id})",
                            EditorStyles.miniLabel
                        );
                    }
                }
            }

            EditorGUILayout.Space(3);

            // Reasoning
            if (!string.IsNullOrEmpty(task.reasoning))
            {
                EditorGUILayout.LabelField("Reasoning:", EditorStyles.boldLabel);
                EditorGUILayout.LabelField(task.reasoning, EditorStyles.wordWrappedMiniLabel);
            }

            EditorGUILayout.Space(5);

            // Action buttons
            EditorGUILayout.BeginHorizontal();

            // Approve & Execute button
            GUI.backgroundColor = Color.green;
            if (GUILayout.Button("✓ Approve & Execute", _buttonStyle, GUILayout.Height(30)))
            {
                // Execute immediately without confirmation dialog
                _manager.ExecuteTask(task);
                _manager.RejectTask(task); // Remove from pending after approval
            }

            GUI.backgroundColor = Color.white;

            // Reject button
            GUI.backgroundColor = new Color(1f, 0.5f, 0.5f);
            if (GUILayout.Button("✗ Reject", _buttonStyle, GUILayout.Height(30)))
            {
                _manager.RejectTask(task);
            }

            GUI.backgroundColor = Color.white;

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        private Color GetComplexityColor(int complexity)
        {
            if (complexity <= 2)
                return Color.green;
            else if (complexity <= 4)
                return Color.yellow;
            else
                return new Color(1f, 0.5f, 0f); // Orange
        }
    }
}
