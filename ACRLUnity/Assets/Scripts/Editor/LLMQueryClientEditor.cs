using UnityEditor;
using UnityEngine;
using PythonCommunication;

namespace EditorScripts
{
    /// <summary>
    /// Custom Unity Inspector editor for LLMQueryClient.
    /// Adds interactive buttons and enhanced UI for querying the RAG/LLM system.
    /// </summary>
    [CustomEditor(typeof(LLMQueryClient))]
    public class LLMQueryClientEditor : Editor
    {
        // Styling
        private GUIStyle _headerStyle;
        private GUIStyle _buttonStyle;
        private GUIStyle _successButtonStyle;
        private GUIStyle _warningButtonStyle;
        private bool _stylesInitialized = false;

        // Foldouts
        private bool _showRecentOperations = false;

        /// <summary>
        /// Initialize custom styles
        /// </summary>
        private void InitializeStyles()
        {
            if (_stylesInitialized)
                return;

            _headerStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 14,
                normal = { textColor = new Color(0.8f, 0.9f, 1.0f) }
            };

            _buttonStyle = new GUIStyle(GUI.skin.button) { fontSize = 12, fixedHeight = 30 };

            _successButtonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fixedHeight = 30,
                normal = { textColor = Color.green },
                fontStyle = FontStyle.Bold
            };

            _warningButtonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fixedHeight = 30,
                normal = { textColor = new Color(1.0f, 0.6f, 0.0f) },
                fontStyle = FontStyle.Bold
            };

            _stylesInitialized = true;
        }

        /// <summary>
        /// Draw custom inspector UI
        /// </summary>
        public override void OnInspectorGUI()
        {
            InitializeStyles();

            LLMQueryClient client = (LLMQueryClient)target;

            // Title
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("🤖 LLM Query Client", _headerStyle);
            EditorGUILayout.LabelField(
                "Send natural language prompts to the RAG/LLM system",
                EditorStyles.miniLabel
            );
            EditorGUILayout.Space(10);

            // Connection Status
            DrawConnectionStatus();
            EditorGUILayout.Space(5);

            // Draw default inspector
            DrawDefaultInspector();

            EditorGUILayout.Space(10);

            // Action Buttons Section
            EditorGUILayout.LabelField("Actions", EditorStyles.boldLabel);

            // Send Query Button (large and prominent)
            GUI.backgroundColor = new Color(0.3f, 0.8f, 0.3f);
            if (GUILayout.Button("📤 Send Query to RAG System", _successButtonStyle))
            {
                client.SendQuery();
            }
            GUI.backgroundColor = Color.white;

            EditorGUILayout.Space(5);

            // Two-column button layout
            EditorGUILayout.BeginHorizontal();

            // Execute Top Operation Button
            GUI.backgroundColor = new Color(0.8f, 0.6f, 0.2f);
            if (GUILayout.Button("🚀 Execute Top Operation", _buttonStyle))
            {
                client.ExecuteTopOperation();
            }
            GUI.backgroundColor = Color.white;

            // Clear Button
            if (GUILayout.Button("🗑️ Clear Prompt", _buttonStyle))
            {
                client.ClearPrompt();
            }

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(10);

            // Quick Action Templates
            DrawQuickActionTemplates(client);

            EditorGUILayout.Space(10);

            // Recent Operations Section
            DrawRecentOperations(client);

            EditorGUILayout.Space(10);

            // Help Section
            DrawHelpSection();
        }

        /// <summary>
        /// Draw connection status indicator
        /// </summary>
        private void DrawConnectionStatus()
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("RAG Server Status:", GUILayout.Width(120));

            if (RAGClient.Instance == null)
            {
                GUI.color = Color.red;
                EditorGUILayout.LabelField("❌ RAGClient not found", EditorStyles.boldLabel);
            }
            else if (RAGClient.Instance.IsConnected)
            {
                GUI.color = Color.green;
                EditorGUILayout.LabelField(
                    $"✓ Connected ({RAGClient.Instance.ConnectionInfo})",
                    EditorStyles.boldLabel
                );
            }
            else
            {
                GUI.color = Color.yellow;
                EditorGUILayout.LabelField("⚠️ Not connected - retrying...", EditorStyles.boldLabel);
            }

            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();
        }

        /// <summary>
        /// Draw quick action template buttons
        /// </summary>
        private void DrawQuickActionTemplates(LLMQueryClient client)
        {
            EditorGUILayout.LabelField("Quick Action Templates", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Move to Position", EditorStyles.miniButton))
            {
                client.Prompt = "Move robot to position x=0.3, y=0.15, z=0.1";
            }

            if (GUILayout.Button("Move Home", EditorStyles.miniButton))
            {
                client.Prompt = "Move robot to home position x=0.0, y=0.0, z=0.3";
            }

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Pick Object", EditorStyles.miniButton))
            {
                client.Prompt = "Pick up object at detected position";
            }

            if (GUILayout.Button("Place Object", EditorStyles.miniButton))
            {
                client.Prompt = "Place object at x=0.2, y=0.0, z=0.1";
            }

            EditorGUILayout.EndHorizontal();
        }

        /// <summary>
        /// Draw recent operations results
        /// </summary>
        private void DrawRecentOperations(LLMQueryClient client)
        {
            _showRecentOperations = EditorGUILayout.BeginFoldoutHeaderGroup(
                _showRecentOperations,
                "Recent Operations"
            );

            if (_showRecentOperations)
            {
                if (client.LastResult == null || client.RecentOperations.Count == 0)
                {
                    EditorGUILayout.LabelField("No operations yet - send a query!", EditorStyles.miniLabel);
                }
                else
                {
                    EditorGUILayout.LabelField(
                        $"Query: {client.LastResult.query}",
                        EditorStyles.wordWrappedLabel
                    );
                    EditorGUILayout.Space(5);

                    for (int i = 0; i < client.RecentOperations.Count; i++)
                    {
                        var op = client.RecentOperations[i];
                        DrawOperationCard(op, i);
                    }
                }
            }

            EditorGUILayout.EndFoldoutHeaderGroup();
        }

        /// <summary>
        /// Draw individual operation card
        /// </summary>
        private void DrawOperationCard(OperationInfo operation, int index)
        {
            EditorGUILayout.BeginVertical(GUI.skin.box);

            // Header
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(
                $"[{index + 1}] {operation.name}",
                EditorStyles.boldLabel,
                GUILayout.Width(200)
            );
            EditorGUILayout.LabelField(
                $"Score: {operation.similarity_score:F3}",
                EditorStyles.miniLabel,
                GUILayout.Width(80)
            );
            EditorGUILayout.LabelField(
                $"Category: {operation.category}",
                EditorStyles.miniLabel
            );
            EditorGUILayout.EndHorizontal();

            // Description
            EditorGUILayout.LabelField(operation.description, EditorStyles.wordWrappedMiniLabel);

            // Parameters
            if (operation.parameters != null && operation.parameters.Length > 0)
            {
                EditorGUILayout.Space(3);
                EditorGUILayout.LabelField("Parameters:", EditorStyles.miniBoldLabel);

                foreach (var param in operation.parameters)
                {
                    string req = param.required ? "required" : "optional";
                    EditorGUILayout.LabelField(
                        $"  • {param.name}: {param.type} ({req})",
                        EditorStyles.miniLabel
                    );
                }
            }

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(3);
        }

        /// <summary>
        /// Draw help section with usage instructions
        /// </summary>
        private void DrawHelpSection()
        {
            EditorGUILayout.BeginVertical(GUI.skin.box);
            EditorGUILayout.LabelField("💡 Help & Usage", EditorStyles.boldLabel);

            EditorGUILayout.LabelField(
                "1. Ensure Python RAGServer is running:",
                EditorStyles.miniLabel
            );
            EditorGUILayout.LabelField(
                "   python -m LLMCommunication.orchestrators.RunRAGServer",
                EditorStyles.miniLabel
            );
            EditorGUILayout.Space(3);

            EditorGUILayout.LabelField(
                "2. Enter a natural language prompt in the 'Prompt' field",
                EditorStyles.miniLabel
            );
            EditorGUILayout.LabelField(
                "3. Click 'Send Query' to search for relevant operations",
                EditorStyles.miniLabel
            );
            EditorGUILayout.LabelField(
                "4. Operations will auto-execute (if enabled) or click 'Execute Top Operation'",
                EditorStyles.miniLabel
            );

            EditorGUILayout.EndVertical();
        }
    }
}
