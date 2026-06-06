using System;
using UnityEngine;

namespace PythonCommunication.Core
{
    /// <summary>
    /// Utility class for parsing JSON with centralized error handling and validation.
    /// Eliminates duplicate try-catch blocks across multiple classes.
    /// </summary>
    public static class JsonParser
    {
        public static bool TryParse<T>(string json, out T result, out string errorMessage)
        {
            result = default(T);
            errorMessage = null;

            if (string.IsNullOrEmpty(json))
            {
                errorMessage = "JSON string is null or empty";
                return false;
            }

            if (json.TrimStart().StartsWith("["))
            {
                errorMessage =
                    "Unity JsonUtility cannot parse top-level arrays. Wrap array in an object.";
                return false;
            }

            try
            {
                result = JsonUtility.FromJson<T>(json);

                if (result == null)
                {
                    errorMessage = $"Unity failed to create object of type {typeof(T).Name}";
                    return false;
                }

                return true;
            }
            catch (ArgumentException ex)
            {
                errorMessage = $"JSON parse error for type {typeof(T).Name}: {ex.Message}";
                return false;
            }
            catch (Exception ex)
            {
                errorMessage =
                    $"Unexpected error parsing JSON for type {typeof(T).Name}: {ex.Message}";
                return false;
            }
        }

        public static bool TryParseWithLogging<T>(string json, out T result, string logPrefix = "")
        {
            bool success = TryParse(json, out result, out string errorMessage);

            if (!success)
            {
                string prefix = string.IsNullOrEmpty(logPrefix) ? "[JsonParser]" : logPrefix;
                Debug.LogError($"{prefix} {errorMessage}");
            }

            return success;
        }
    }
}
