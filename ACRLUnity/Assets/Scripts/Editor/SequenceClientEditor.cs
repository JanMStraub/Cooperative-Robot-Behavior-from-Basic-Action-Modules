using PythonCommunication;
using UnityEditor;
using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// Custom Unity Inspector editor for SequenceClient.
    /// Adds interactive buttons and enhanced UI for sending compound command sequences.
    /// </summary>
    [CustomEditor(typeof(SequenceClient))]
    public class SequenceClientEditor : Editor
    {
        // Styling
        private GUIStyle _subHeaderStyle;
        private GUIStyle _buttonStyle;
        private GUIStyle _successButtonStyle;
        private GUIStyle _warningButtonStyle;
        private GUIStyle _boxStyle;
        private GUIStyle _statusBoxStyle;
        private bool _stylesInitialized = false;

        // Foldouts
        private bool _showRecentCommands = false;
        private bool _showLastResult = true;
        private bool _showQuickActions = true;

        // Colors
        private readonly Color _successColor = new Color(0.2f, 0.8f, 0.2f);
        private readonly Color _warningColor = new Color(1.0f, 0.7f, 0.0f);
        private readonly Color _errorColor = new Color(0.9f, 0.3f, 0.3f);
        private readonly Color _infoColor = new Color(0.4f, 0.7f, 1.0f);

        /// <summary>
        /// Initialize custom styles
        /// </summary>
        private void InitializeStyles()
        {
            if (_stylesInitialized && _boxStyle != null)
                return;

            _subHeaderStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 12,
                normal = { textColor = new Color(0.7f, 0.7f, 0.7f) },
                padding = new RectOffset(5, 5, 5, 5),
            };

            _buttonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fixedHeight = 20,
                margin = new RectOffset(5, 5, 5, 5),
            };

            _successButtonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fixedHeight = 20,
                fontStyle = FontStyle.Bold,
                margin = new RectOffset(5, 5, 5, 5),
            };

            _warningButtonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fixedHeight = 20,
                margin = new RectOffset(5, 5, 5, 5),
            };

            _boxStyle = new GUIStyle(EditorStyles.helpBox)
            {
                padding = new RectOffset(5, 5, 5, 5),
                margin = new RectOffset(0, 0, 5, 5),
            };

            _statusBoxStyle = new GUIStyle(EditorStyles.helpBox)
            {
                padding = new RectOffset(5, 5, 5, 5),
                margin = new RectOffset(0, 0, 3, 3),
            };

            _stylesInitialized = true;
        }

        /// <summary>
        /// Draw custom inspector UI
        /// </summary>
        public override void OnInspectorGUI()
        {
            InitializeStyles();

            SequenceClient client = (SequenceClient)target;

            // Connection Status
            DrawConnectionStatus(client);

            EditorGUILayout.Space(5);

            // Default Inspector (settings)
            DrawSection(() =>
            {
                DrawDefaultInspector();
            });

            EditorGUILayout.Space(5);

            // Action Buttons Section
            DrawSection(() =>
            {
                EditorGUILayout.LabelField("Actions", _subHeaderStyle);
                EditorGUILayout.Space(5);

                EditorGUILayout.BeginHorizontal();
                GUI.backgroundColor = _successColor;
                if (GUILayout.Button("Send Prompt", _successButtonStyle))
                {
                    client.SendSequence();
                }
                
                GUI.backgroundColor = _warningColor;
                if (GUILayout.Button("Clear Prompt", _warningButtonStyle))
                {
                    client.ClearPrompt();
                }
                GUI.backgroundColor = Color.white;

                EditorGUILayout.EndHorizontal();
            });

            EditorGUILayout.Space(5);

            // Quick Action Templates
            _showQuickActions = EditorGUILayout.BeginFoldoutHeaderGroup(
                _showQuickActions,
                "Quick Action Templates"
            );
            if (_showQuickActions)
            {
                DrawQuickActionTemplates(client);
            }
            EditorGUILayout.EndFoldoutHeaderGroup();

            EditorGUILayout.Space(5);

            // Last Result Section
            _showLastResult = EditorGUILayout.BeginFoldoutHeaderGroup(
                _showLastResult,
                "Last Sequence Result"
            );
            if (_showLastResult)
            {
                DrawLastResult(client);
            }
            EditorGUILayout.EndFoldoutHeaderGroup();

            EditorGUILayout.Space(5);

            // Recent Commands Section
            _showRecentCommands = EditorGUILayout.BeginFoldoutHeaderGroup(
                _showRecentCommands,
                "Recent Commands"
            );
            if (_showRecentCommands)
            {
                DrawRecentCommands(client);
            }
            EditorGUILayout.EndFoldoutHeaderGroup();

            EditorGUILayout.Space(10);

            // Force repaint for real-time updates
            if (Application.isPlaying)
            {
                Repaint();
            }
        }

        /// <summary>
        /// Draw a section with background
        /// </summary>
        private void DrawSection(System.Action content)
        {
            EditorGUILayout.BeginVertical(_boxStyle);
            content();
            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Draw connection status indicator
        /// </summary>
        private void DrawConnectionStatus(SequenceClient client)
        {
            EditorGUILayout.BeginVertical(_statusBoxStyle);
            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.LabelField("Status:", EditorStyles.boldLabel, GUILayout.Width(60));

            if (client == null)
            {
                GUI.color = _errorColor;
                EditorGUILayout.LabelField("Client not found", EditorStyles.boldLabel);
            }
            else if (client.IsConnected)
            {
                GUI.color = _successColor;
                EditorGUILayout.LabelField(
                    $"Connected ({client.ConnectionInfo})",
                    EditorStyles.boldLabel
                );
            }
            else
            {
                GUI.color = _warningColor;
                EditorGUILayout.LabelField("Not connected - retrying...", EditorStyles.boldLabel);
            }

            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Draw quick action template buttons
        /// </summary>
        private void DrawQuickActionTemplates(SequenceClient client)
        {
            EditorGUILayout.BeginVertical(_boxStyle);

            // Basic Actions
            EditorGUILayout.LabelField("Basic Actions", EditorStyles.miniBoldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Move to Position", _buttonStyle))
            {
                client.Prompt = "Move robot to position x=0, y=0.3, z=0";
            }
            if (GUILayout.Button("Start Position", _buttonStyle))
            {
                client.Prompt = "Move the robot to the start position";
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Open Gripper", _buttonStyle))
            {
                client.Prompt = "Open gripper";
            }
            if (GUILayout.Button("Close Gripper", _buttonStyle))
            {
                client.Prompt = "Close gripper";
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Compound Actions
            EditorGUILayout.LabelField("Compound Actions", EditorStyles.miniBoldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Move & Close Grip", _buttonStyle))
            {
                client.Prompt = "move to x=0, y=0.3, z=0 and close the gripper";
            }
            if (GUILayout.Button("Move & Open Grip", _buttonStyle))
            {
                client.Prompt = "move to x=0, y=0.3, z=0 and open the gripper";
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Pick & Place
            EditorGUILayout.LabelField("Pick & Place Sequences", EditorStyles.miniBoldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Pick Sequence", _buttonStyle))
            {
                client.Prompt =
                    "move to (0.3, 0.15, 0.05), then close the gripper, then move to (0.3, 0.15, 0.2)";
            }
            if (GUILayout.Button("Place Sequence", _buttonStyle))
            {
                client.Prompt =
                    "move to (0.1, 0.3, 0.2), then move to (0.1, 0.3, 0.05), then open the gripper, then move to (0.1, 0.3, 0.2)";
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Perception
            EditorGUILayout.LabelField("Perception Commands", EditorStyles.miniBoldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Find Blue Cube", _buttonStyle))
            {
                client.Prompt = "Calculate the coordinates of the blue cube on the left";
            }
            if (GUILayout.Button("Pick at Object", _buttonStyle))
            {
                client.Prompt = "Pick up object at detected position";
            }
            if (GUILayout.Button("Place Object", _buttonStyle))
            {
                client.Prompt = "Place object at x=0.2, y=0.0, z=0.1";
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Draw last result display
        /// </summary>
        private void DrawLastResult(SequenceClient client)
        {
            EditorGUILayout.BeginVertical(_boxStyle);

            if (client.LastResult == null)
            {
                EditorGUILayout.LabelField(
                    "No results yet - send a sequence!",
                    EditorStyles.centeredGreyMiniLabel
                );
            }
            else
            {
                var result = client.LastResult;

                // Status with colored background
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("Status:", GUILayout.Width(60));

                if (result.success)
                {
                    GUI.color = _successColor;
                    EditorGUILayout.LabelField("SUCCESS", EditorStyles.boldLabel);
                }
                else
                {
                    GUI.color = _errorColor;
                    EditorGUILayout.LabelField("FAILED", EditorStyles.boldLabel);
                }
                GUI.color = Color.white;
                EditorGUILayout.EndHorizontal();

                // Stats row
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField(
                    $"Commands: {result.completed_commands}/{result.total_commands}",
                    GUILayout.Width(120)
                );
                EditorGUILayout.LabelField($"Duration: {result.total_duration_ms:F0}ms");
                EditorGUILayout.EndHorizontal();

                // Error if any
                if (!string.IsNullOrEmpty(result.error))
                {
                    EditorGUILayout.Space(5);
                    GUI.color = _errorColor;
                    EditorGUILayout.LabelField(
                        $"Error: {result.error}",
                        EditorStyles.wordWrappedMiniLabel
                    );
                    GUI.color = Color.white;
                }

                // Individual command results
                if (result.results != null && result.results.Count > 0)
                {
                    EditorGUILayout.Space(8);
                    EditorGUILayout.LabelField("Command Details:", EditorStyles.miniBoldLabel);

                    foreach (var cmdResult in result.results)
                    {
                        EditorGUILayout.BeginHorizontal();

                        // Status indicator
                        if (cmdResult.success)
                        {
                            GUI.color = _successColor;
                            GUILayout.Label("●", GUILayout.Width(15));
                        }
                        else
                        {
                            GUI.color = _errorColor;
                            GUILayout.Label("●", GUILayout.Width(15));
                        }
                        GUI.color = Color.white;

                        // Operation name and duration
                        EditorGUILayout.LabelField(
                            $"{cmdResult.operation}",
                            EditorStyles.miniLabel,
                            GUILayout.ExpandWidth(true)
                        );
                        EditorGUILayout.LabelField(
                            $"{cmdResult.duration_ms:F0}ms",
                            EditorStyles.miniLabel,
                            GUILayout.Width(50)
                        );

                        EditorGUILayout.EndHorizontal();

                        // Show error if failed
                        if (!string.IsNullOrEmpty(cmdResult.error))
                        {
                            GUI.color = _errorColor;
                            EditorGUILayout.LabelField(
                                $"  └ {cmdResult.error}",
                                EditorStyles.miniLabel
                            );
                            GUI.color = Color.white;
                        }
                    }
                }
            }

            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Draw recent commands list
        /// </summary>
        private void DrawRecentCommands(SequenceClient client)
        {
            EditorGUILayout.BeginVertical(_boxStyle);

            if (client.RecentCommands == null || client.RecentCommands.Count == 0)
            {
                EditorGUILayout.LabelField(
                    "No recent commands",
                    EditorStyles.centeredGreyMiniLabel
                );
            }
            else
            {
                for (int i = 0; i < client.RecentCommands.Count; i++)
                {
                    EditorGUILayout.BeginHorizontal();

                    // Command number
                    GUI.color = _infoColor;
                    EditorGUILayout.LabelField($"{i + 1}.", GUILayout.Width(20));
                    GUI.color = Color.white;

                    // Command text (truncated if too long)
                    string cmdText = client.RecentCommands[i];
                    if (cmdText.Length > 50)
                    {
                        cmdText = cmdText.Substring(0, 47) + "...";
                    }
                    EditorGUILayout.LabelField(cmdText, EditorStyles.miniLabel);

                    // Use button
                    if (GUILayout.Button("Use", GUILayout.Width(40)))
                    {
                        client.Prompt = client.RecentCommands[i];
                    }

                    EditorGUILayout.EndHorizontal();
                }
            }

            EditorGUILayout.EndVertical();
        }
    }
}
