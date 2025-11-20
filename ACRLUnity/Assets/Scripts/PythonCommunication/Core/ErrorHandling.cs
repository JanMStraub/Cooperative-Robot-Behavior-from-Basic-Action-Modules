using System;

namespace PythonCommunication.Core
{
    /// <summary>
    /// Standardized error codes for communication operations.
    /// Provides consistent error taxonomy across Unity-Python boundary.
    /// </summary>
    public enum ErrorCode
    {
        // Success
        SUCCESS = 0,

        // Network errors (1000-1999)
        NETWORK_ERROR = 1000,
        CONNECTION_CLOSED = 1001,
        CONNECTION_TIMEOUT = 1002,
        CONNECTION_REFUSED = 1003,
        INVALID_RESPONSE = 1004,

        // Protocol errors (2000-2999)
        PROTOCOL_ERROR = 2000,
        INVALID_MESSAGE_TYPE = 2001,
        INVALID_MESSAGE_LENGTH = 2002,
        INCOMPLETE_MESSAGE = 2003,
        PROTOCOL_VERSION_MISMATCH = 2004,

        // Validation errors (3000-3999)
        VALIDATION_ERROR = 3000,
        INVALID_PARAMETER = 3001,
        MISSING_PARAMETER = 3002,
        PARAMETER_OUT_OF_RANGE = 3003,

        // Parsing errors (4000-4999)
        PARSE_ERROR = 4000,
        JSON_PARSE_ERROR = 4001,
        ENCODING_ERROR = 4002,

        // Operation errors (5000-5999)
        OPERATION_FAILED = 5000,
        TIMEOUT = 5001,
        ROBOT_NOT_FOUND = 5002,
        OPERATION_NOT_SUPPORTED = 5003,

        // Internal errors (6000-6999)
        INTERNAL_ERROR = 6000,
        THREAD_ERROR = 6001,
        INITIALIZATION_ERROR = 6002,

        // Unknown
        UNKNOWN_ERROR = 9999
    }

    /// <summary>
    /// Detailed error information for operations.
    /// </summary>
    [Serializable]
    public class ErrorInfo
    {
        public ErrorCode code;
        public string message;
        public string details;

        public ErrorInfo(ErrorCode code, string message, string details = null)
        {
            this.code = code;
            this.message = message;
            this.details = details;
        }

        public override string ToString()
        {
            string result = $"[{code}] {message}";
            if (!string.IsNullOrEmpty(details))
            {
                result += $" | {details}";
            }
            return result;
        }
    }

    /// <summary>
    /// Result wrapper for operations that can fail.
    /// Provides structured success/failure with error details.
    /// </summary>
    /// <typeparam name="T">Type of successful result</typeparam>
    public class Result<T>
    {
        public bool Success { get; private set; }
        public T Value { get; private set; }
        public ErrorInfo Error { get; private set; }

        private Result(bool success, T value, ErrorInfo error)
        {
            Success = success;
            Value = value;
            Error = error;
        }

        /// <summary>
        /// Create a successful result
        /// </summary>
        public static Result<T> Ok(T value)
        {
            return new Result<T>(true, value, null);
        }

        /// <summary>
        /// Create a failed result
        /// </summary>
        public static Result<T> Fail(ErrorCode code, string message, string details = null)
        {
            return new Result<T>(false, default(T), new ErrorInfo(code, message, details));
        }

        /// <summary>
        /// Create a failed result from ErrorInfo
        /// </summary>
        public static Result<T> Fail(ErrorInfo error)
        {
            return new Result<T>(false, default(T), error);
        }

        /// <summary>
        /// Match pattern for handling success/failure
        /// </summary>
        public TResult Match<TResult>(Func<T, TResult> onSuccess, Func<ErrorInfo, TResult> onFailure)
        {
            return Success ? onSuccess(Value) : onFailure(Error);
        }

        /// <summary>
        /// Execute action based on success/failure
        /// </summary>
        public void Match(Action<T> onSuccess, Action<ErrorInfo> onFailure)
        {
            if (Success)
            {
                onSuccess(Value);
            }
            else
            {
                onFailure(Error);
            }
        }
    }

    /// <summary>
    /// Result wrapper for operations without return value
    /// </summary>
    public class Result
    {
        public bool Success { get; private set; }
        public ErrorInfo Error { get; private set; }

        private Result(bool success, ErrorInfo error)
        {
            Success = success;
            Error = error;
        }

        /// <summary>
        /// Create a successful result
        /// </summary>
        public static Result Ok()
        {
            return new Result(true, null);
        }

        /// <summary>
        /// Create a failed result
        /// </summary>
        public static Result Fail(ErrorCode code, string message, string details = null)
        {
            return new Result(false, new ErrorInfo(code, message, details));
        }

        /// <summary>
        /// Create a failed result from ErrorInfo
        /// </summary>
        public static Result Fail(ErrorInfo error)
        {
            return new Result(false, error);
        }

        /// <summary>
        /// Execute action based on success/failure
        /// </summary>
        public void Match(Action onSuccess, Action<ErrorInfo> onFailure)
        {
            if (Success)
            {
                onSuccess();
            }
            else
            {
                onFailure(Error);
            }
        }
    }
}
