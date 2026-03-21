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

        // Original GUI colors captured at the start of each OnInspectorGUI call
        private Color _originalColor;
        private Color _originalBgColor;

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
        /// Drive constant repaints while in Play Mode for real-time updates,
        /// without busy-looping inside OnInspectorGUI.
        /// </summary>
        public override bool RequiresConstantRepaint() => Application.isPlaying;

        /// <summary>
        /// Draw custom inspector UI
        /// </summary>
        public override void OnInspectorGUI()
        {
            InitializeStyles();

            _originalColor = GUI.color;
            _originalBgColor = GUI.backgroundColor;

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
                    // Defer past the current GUI frame to avoid corrupting the
                    // GUILayout state if a response arrives and triggers a repaint
                    // before EndHorizontal/EndVertical have been called.
                    EditorApplication.delayCall += () => client.SendSequence();
                }

                GUI.backgroundColor = _warningColor;
                if (GUILayout.Button("Clear Prompt", _warningButtonStyle))
                {
                    EditorApplication.delayCall += () => client.ClearPrompt();
                }
                GUI.backgroundColor = _originalBgColor;

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

            GUI.color = _originalColor;
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
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Move robot 1 to position x=0, y=0.3, z=0";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Start Position", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Move robot 1 to the start position";
                EditorUtility.SetDirty(client);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Open Gripper", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot 1: Open gripper";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Close Gripper", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot1: Close gripper";
                EditorUtility.SetDirty(client);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Estimate Distance", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Calculate the distance between the blue and the red cube";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Move from a to b", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Move robot 1 from x=0, y=0.3, z=0 to x=0.1, y=0.1, z=0.1";
                EditorUtility.SetDirty(client);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Rotate robot", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Rotate the gripper of robot 2 90 degrees";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Pick object from position", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot 1: Pick the object from (-0.2, 0, 0.05)";
                EditorUtility.SetDirty(client);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Release object", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot 1: Release object";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Pick object from position", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Pick the object from (-0.2, 0, 0.05)";
                EditorUtility.SetDirty(client);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Pick & Place
            EditorGUILayout.LabelField("Pick & Place Sequences", EditorStyles.miniBoldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Pick Sequence", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot 1: Grab the blue cube on the left";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Place Sequence", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot 1: Move to field g and place the object there";
                EditorUtility.SetDirty(client);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Cooperation
            EditorGUILayout.LabelField("Cooperation Commands", EditorStyles.miniBoldLabel);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Transfer cube", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt =
                    @"PARALLEL GROUP 1:
- Robot1: Grab red cube

PARALLEL GROUP 2:
- Robot1: Move to coordinate x=-0.1, y=0.5, z=0.0
- Robot1: Signal 'object_ready_for_handoff'
- Robot2: Wait for signal 'object_ready_for_handoff'

PARALLEL GROUP 3:
- Robot2: Grab red cube
- Robot2: Signal 'handoff_complete'
- Robot1: Wait for signal 'handoff_complete'

PARALLEL GROUP 4:
- Robot1: Open gripper
- Robot1: Move to start position
- Robot2: Move to start position";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("Transfer cube short", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "Robot1 and Robot2 perform a handoff of the red cube";
                EditorUtility.SetDirty(client);
            }
            if (GUILayout.Button("None", _buttonStyle))
            {
                Undo.RecordObject(client, "Change Prompt");
                client.Prompt = "";
                EditorUtility.SetDirty(client);
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
                GUI.color = _originalColor;
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
                    GUI.color = _originalColor;
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
                        GUI.color = _originalColor;

                        // Operation name and duration
                        EditorGUILayout.LabelField(
                            cmdResult.operation,
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
                            GUI.color = _originalColor;
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
                    GUI.color = _originalColor;

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
                        Undo.RecordObject(client, "Change Prompt");
                        client.Prompt = client.RecentCommands[i];
                        EditorUtility.SetDirty(client);
                    }

                    EditorGUILayout.EndHorizontal();
                }
            }

            EditorGUILayout.EndVertical();
        }
    }
}
