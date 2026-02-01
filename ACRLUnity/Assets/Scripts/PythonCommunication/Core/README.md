# Unity-Python Communication Core Architecture

## Overview

The `ACRLUnity/Assets/Scripts/PythonCommunication/Core` folder contains the foundational infrastructure for TCP-based communication between Unity and Python backend servers. This system implements **Protocol V2**, a robust wire protocol with request/response correlation to prevent race conditions in multi-robot scenarios.

The core architecture consists of four files that work together to provide a clean, reusable foundation for all network communication:

1. **TCPClientBase.cs** - Abstract base for TCP client connections
2. **BidirectionalClientBase.cs** - Abstract base for request/response clients
3. **UnityProtocol.cs** - Wire protocol V2 message encoding/decoding
4. **JsonParser.cs** - Centralized JSON parsing with error handling

---

## 1. TCPClientBase.cs

### Purpose
Provides common TCP connection management functionality that all network clients inherit. This eliminates duplicate code across multiple client implementations (ImageSender, UnifiedPythonReceiver, SequenceClient, etc.).

### Key Features

#### Connection Management
- **Asynchronous connection** (`TCPClientBase.cs:155-174`): Non-blocking connection to avoid freezing Unity's main thread
- **Auto-reconnect** (`TCPClientBase.cs:114-126`): Automatically attempts reconnection every 2 seconds on connection loss
- **Thread-safe state management** (`TCPClientBase.cs:44-46`): Uses `_connectionLock` and `_writeLock` for concurrent access
- **Proper cleanup** (`TCPClientBase.cs:307-370`): Gracefully closes connections on application quit or component destruction

#### Thread Safety
The class uses **SynchronizationContext** (`TCPClientBase.cs:51`) to marshal callbacks from background threads to Unity's main thread:

```csharp
_mainThreadContext.Post(_ => {
    OnConnected();  // Safe to call Unity API here
}, null);
```

This is critical because Unity's API is not thread-safe and can only be called from the main thread.

#### Lifecycle Hooks
Provides virtual methods that subclasses can override to implement custom behavior:

- `OnConnected()` (`TCPClientBase.cs:383`): Called when connection succeeds (on main thread)
- `OnConnectionFailed()` (`TCPClientBase.cs:393`): Called when connection fails (on main thread)
- `OnDisconnecting()` (`TCPClientBase.cs:402`): Called before disconnecting (calling thread)
- `OnDisconnected()` (`TCPClientBase.cs:411`): Called after disconnecting (calling thread)

#### Utility Methods

**ReadExactly()** (`TCPClientBase.cs:433-448`): Blocking read that ensures exactly N bytes are read from the stream. Critical for reading length-prefixed messages:

```csharp
// Read exactly 4 bytes for message length
ReadExactly(stream, buffer, 4);
```

**WriteToStream()** (`TCPClientBase.cs:457-492`): Thread-safe write with proper locking to prevent message corruption:

```csharp
lock (_writeLock) {
    streamCopy.Write(data, 0, data.Length);
    streamCopy.Flush();  // Atomic write + flush
}
```

**GenerateRequestId()** (`TCPClientBase.cs:502-512`): Thread-safe counter for Protocol V2 request correlation. Generates unique IDs across all messages, wrapping at `uint.MaxValue`.

### Connection State Properties

- `IsConnected` (`TCPClientBase.cs:62-82`): Property that reflects connection state. **Warning**: This is not real-time - it only reflects the state as of the last I/O operation.
- `ConnectionInfo` (`TCPClientBase.cs:87`): Returns formatted connection string (`host:port`) for logging.

### Unity Lifecycle Integration

The class integrates with Unity's component lifecycle:

- **Awake()** (`TCPClientBase.cs:94-98`): Captures main thread context
- **Start()** (`TCPClientBase.cs:103-109`): Auto-connects if enabled
- **Update()** (`TCPClientBase.cs:114-126`): Handles auto-reconnect logic
- **OnApplicationQuit()/OnDestroy()** (`TCPClientBase.cs:131-146`): Cleanup

---

## 2. BidirectionalClientBase.cs

### Purpose
Extends `TCPClientBase` to support bidirectional request/response communication patterns. Implements request ID correlation, background receive thread, and main-thread callback dispatch.

### Generic Type Parameter
The class is generic over the response type `TResponse`:

```csharp
public abstract class BidirectionalClientBase<TResponse> : TCPClientBase
    where TResponse : class
```

This allows different clients to handle different response types (e.g., `DetectionResult`, `SequenceResponse`, etc.).

### Key Features

#### Background Receive Thread
When connected, the class spawns a dedicated background thread (`BidirectionalClientBase.cs:49-60`) that continuously receives responses:

```csharp
protected override void OnConnected() {
    _receiveShouldRun = true;
    _receiveThread = new Thread(ReceiveLoop);
    _receiveThread.Start();
}
```

The **ReceiveLoop()** (`BidirectionalClientBase.cs:91-134`) runs on the background thread:
1. Calls abstract `ReceiveResponse()` to read and parse the next response
2. Enqueues the response in a thread-safe queue
3. Handles exceptions and triggers disconnection on errors

#### Request/Response Correlation
The class maintains a dictionary of pending requests (`BidirectionalClientBase.cs:29-30`):

```csharp
protected Dictionary<uint, Action<TResponse>> _pendingRequests;
```

When sending a request with a callback:

```csharp
SendRequest(data, requestId, callback: response => {
    // This callback will be invoked when the response with matching requestId arrives
});
```

The correlation happens in `ProcessResponseQueue()` (`BidirectionalClientBase.cs:160-213`):
1. Extract `request_id` from the response (via `GetResponseRequestId()`)
2. Look up the matching callback in `_pendingRequests`
3. Invoke the callback with the response
4. Call the general `OnResponseReceived()` hook for all responses

#### Main Thread Processing
The `ProcessResponseQueue()` method (`BidirectionalClientBase.cs:160-213`) runs in Unity's `Update()` loop:
- Dequeues responses from the background thread's queue
- **Limits to 50 items per frame** (`BidirectionalClientBase.cs:33`) to prevent frame drops
- Invokes callbacks safely on the main thread

#### Utility Methods

**ReadUInt32BE()** (`BidirectionalClientBase.cs:292-301`): Reads a 4-byte big-endian unsigned integer. Handles endianness conversion for cross-platform compatibility:

```csharp
if (BitConverter.IsLittleEndian)
    Array.Reverse(buffer);  // Convert to big-endian
```

**ReadString()** (`BidirectionalClientBase.cs:306-324`): Reads a length-prefixed UTF-8 string:
1. Read 4-byte length
2. Validate length (max 10MB)
3. Read N bytes
4. Decode as UTF-8

**WriteUInt32BE()** (`BidirectionalClientBase.cs:337-343`): Writes a 4-byte big-endian integer to a buffer.

### Abstract Methods

Subclasses must implement:

- `ReceiveResponse()` (`BidirectionalClientBase.cs:141`): Parse a single response from the network stream
- `LogPrefix` (`BidirectionalClientBase.cs:35`): Property for logging identification (e.g., `"[SEQUENCE_CLIENT]"`)
- `GetResponseRequestId()` (`BidirectionalClientBase.cs:147`): Extract request ID from response (optional, defaults to 0)

---

## 3. UnityProtocol.cs

### Purpose
Implements the **Protocol V2 wire format** for all Unity ↔ Python messages. This is a stateless utility class with encoding/decoding methods for different message types.

### Protocol V2 Overview

All messages include a **5-byte header**:

```
[message_type:1 byte][request_id:4 bytes]
```

- **message_type**: Identifies the message type (IMAGE, RESULT, RAG_QUERY, etc.)
- **request_id**: Unsigned 32-bit integer for correlating requests with responses

This design eliminates race conditions where responses arrive out of order or get mismatched with requests.

### Message Type Enumeration (`UnityProtocol.cs:10-19`)

```csharp
public enum MessageType : byte {
    IMAGE = 0x01,           // Single camera image (Unity → Python)
    RESULT = 0x02,          // JSON result (Python → Unity)
    RAG_QUERY = 0x03,       // RAG query (Unity → Python)
    RAG_RESPONSE = 0x04,    // RAG response (Python → Unity)
    STATUS_QUERY = 0x05,    // Robot status query (Unity → Python)
    STATUS_RESPONSE = 0x06, // Robot status response (bidirectional)
    STEREO_IMAGE = 0x07,    // Stereo image pair (Unity → Python)
}
```

**Important**: This enum must match exactly with the Python implementation in `ACRLPython/core/UnityProtocol.py`.

### Constants (`UnityProtocol.cs:33-37`)

```csharp
public const int VERSION = 2;
public const int INT_SIZE = 4;
public const int TYPE_SIZE = 1;
public const int HEADER_SIZE = 5;  // TYPE_SIZE + INT_SIZE
public const int MAX_IMAGE_SIZE = 10 * 1024 * 1024;  // 10MB
```

### Header Encoding/Decoding

**EncodeHeader()** (`UnityProtocol.cs:49-55`): Creates the 5-byte header
```csharp
byte[] header = new byte[5];
header[0] = (byte)messageType;
Buffer.BlockCopy(BitConverter.GetBytes(requestId), 0, header, 1, 4);
```

**DecodeHeader()** (`UnityProtocol.cs:65-86`): Parses the header from incoming data
```csharp
messageType = (MessageType)data[offset];
requestId = BitConverter.ToUInt32(data, offset + 1);
```

### Image Messages (Unity → Python)

**EncodeImageMessage()** (`UnityProtocol.cs:101-180`): Single camera image

Message format:
```
[type:1][request_id:4]
[camera_id_len:4][camera_id:N]
[prompt_len:4][prompt:N]
[image_len:4][image_data:N]
```

Example usage:
```csharp
byte[] message = UnityProtocol.EncodeImageMessage(
    cameraId: "FrontCamera",
    prompt: "Detect red cubes",
    imageBytes: pngData,
    requestId: 12345
);
```

**EncodeStereoImageMessage()** (`UnityProtocol.cs:195-343`): Stereo image pair

Message format:
```
[type:1][request_id:4]
[pair_id_len:4][pair_id:N]
[cam_L_id_len:4][cam_L_id:N]
[cam_R_id_len:4][cam_R_id:N]
[prompt_len:4][prompt:N]
[img_L_len:4][img_L:N]
[img_R_len:4][img_R:N]
```

### Result Messages (Python → Unity)

**DecodeResultMessage()** (`UnityProtocol.cs:395-425`): JSON result from Python

Message format:
```
[type:1][request_id:4]
[json_len:4][json_data:N]
```

Example:
```csharp
string jsonResult = UnityProtocol.DecodeResultMessage(data, out uint requestId);
// requestId can be used to correlate with the original request
```

**EncodeResultMessage()** (`UnityProtocol.cs:434-463`): For testing/sending results back to Python

### RAG Query/Response Messages

**EncodeRagQuery()** (`UnityProtocol.cs:478-540`): Semantic operation search

Message format:
```
[type:1][request_id:4]
[query_len:4][query_text:N]
[top_k:4]
[filters_json_len:4][filters_json:N]
```

Example:
```csharp
byte[] query = UnityProtocol.EncodeRagQuery(
    query: "move to the red cube",
    topK: 5,
    filtersJson: "{}",
    requestId: requestId
);
```

**DecodeRagResponse()** (`UnityProtocol.cs:586-611`): Operation context from RAG system

### Status Query/Response Messages

**EncodeStatusQuery()** (`UnityProtocol.cs:653-689`): Robot status request

Message format:
```
[type:1][request_id:4]
[robot_id_len:4][robot_id:N]
[detailed:1]
```

**EncodeStatusResponse()** (`UnityProtocol.cs:764-785`): Robot status JSON response

Message format:
```
[type:1][request_id:4]
[json_len:4][robot_status_json:N]
```

### Validation Helpers

**PeekMessageType()** (`UnityProtocol.cs:804-812`): Read message type without decoding
```csharp
MessageType type = UnityProtocol.PeekMessageType(data);
```

**PeekRequestId()** (`UnityProtocol.cs:817-825`): Read request ID without full decode

**IsValidImageSize()** (`UnityProtocol.cs:794-799`): Validate image is within limits

---

## 4. JsonParser.cs

### Purpose
Centralized JSON parsing with comprehensive error handling and validation. Eliminates duplicate try-catch blocks across multiple classes.

### Key Features

#### Safe Parsing with TryParse Pattern

**TryParse()** (`JsonParser.cs:20-60`): The core parsing method
```csharp
if (JsonParser.TryParse<DetectionResult>(json, out var result, out string error)) {
    // Use result
} else {
    Debug.LogError($"Parse failed: {error}");
}
```

Error handling includes:
1. Null/empty string detection
2. Top-level array validation (Unity's JsonUtility can't parse arrays)
3. Type validation
4. Exception handling with detailed error messages

#### Logging Variant

**TryParseWithLogging()** (`JsonParser.cs:70-81`): Automatically logs errors
```csharp
if (JsonParser.TryParseWithLogging<Config>(json, out var config, "[CONFIG]")) {
    // Config is valid
}
// Error was already logged if parsing failed
```

#### Throwing Variant

**Parse()** (`JsonParser.cs:91-99`): Throws `JsonParseException` on failure
```csharp
try {
    var result = JsonParser.Parse<DetectionResult>(json);
} catch (JsonParseException ex) {
    // Handle critical parsing failure
}
```

### Custom Exception

**JsonParseException** (`JsonParser.cs:105-112`): Specific exception type for JSON parsing errors, making it easy to distinguish from other exceptions.

---

## How The Components Work Together

### Example: Sending a Command with Response

Here's how the components collaborate in a typical request/response scenario:

```csharp
// 1. SequenceClient (inherits BidirectionalClientBase) prepares a command
string command = "move to cube";
uint requestId = GenerateRequestId();  // From TCPClientBase

// 2. Encode using UnityProtocol
byte[] message = UnityProtocol.EncodeRagQuery(
    query: command,
    topK: 5,
    requestId: requestId
);

// 3. Send with callback registration (BidirectionalClientBase)
SendRequest(message, requestId, response => {
    // This callback will be invoked on main thread when response arrives
    Debug.Log($"Command completed: {response}");
});

// Meanwhile, on background thread (BidirectionalClientBase):
// 4. ReceiveLoop() reads response data
TResponse response = ReceiveResponse();  // Implemented by subclass

// 5. UnityProtocol decodes the message
string json = UnityProtocol.DecodeRagResponse(data, out uint responseRequestId);

// 6. JsonParser parses the JSON
if (JsonParser.TryParse<OperationContext>(json, out var context, out string error)) {
    // Queue for main thread processing
    lock (_queueLock) {
        _responseQueue.Enqueue(context);
    }
}

// Back on Unity main thread in Update():
// 7. ProcessResponseQueue() matches request_id and invokes callback
Action<TResponse> callback = _pendingRequests[responseRequestId];
callback(response);  // User's callback from step 3
```

### Inheritance Hierarchy

```
MonoBehaviour
    └─ TCPClientBase (abstract)
           ├─ Connection management
           ├─ Thread safety
           └─ Utility methods
                  │
                  └─ BidirectionalClientBase<TResponse> (abstract)
                         ├─ Background receive thread
                         ├─ Request/response correlation
                         └─ Main thread callback dispatch
                                │
                                ├─ SequenceClient (concrete)
                                ├─ UnifiedPythonReceiver (concrete)
                                └─ (Other client implementations)
```

### Data Flow Diagram

```
Unity Game Thread                Background Thread                Python Server
─────────────────                ─────────────────                ─────────────

SendRequest() ─────────┐
                       │
GenerateRequestId()    │
                       │
UnityProtocol.Encode() │
                       │
WriteToStream() ──────>├──────────────────────────────────────> Server processes
                       │                                         request
                       │
                       │                    <──────────────────  Server sends
                       │                                         response
                       │
                       │         ReceiveLoop()
                       │         UnityProtocol.Decode()
                       │         JsonParser.TryParse()
                       │         Enqueue(response)
                       │                   │
Update() ─────────────>├──────────────────┘
ProcessResponseQueue() │
Match request_id       │
Invoke callback        │
OnResponseReceived()   │
```

---

## Design Patterns and Best Practices

### 1. Template Method Pattern
`TCPClientBase` and `BidirectionalClientBase` use the template method pattern, providing a framework that subclasses extend with specific behavior:

```csharp
// Base class provides the framework
protected virtual void ReceiveLoop() {
    while (_receiveShouldRun) {
        TResponse response = ReceiveResponse();  // Subclass implements
        EnqueueResponse(response);
    }
}

// Subclass provides specific implementation
protected override TResponse ReceiveResponse() {
    // Read protocol-specific message format
}
```

### 2. Thread Safety
Multiple synchronization primitives ensure thread-safe operation:

- `_connectionLock` (`TCPClientBase.cs:44`): Protects connection state
- `_writeLock` (`TCPClientBase.cs:45`): Prevents message interleaving during writes
- `_queueLock` (`BidirectionalClientBase.cs:24`): Protects response queue
- `_pendingLock` (`BidirectionalClientBase.cs:31`): Protects pending requests dictionary

### 3. Resource Cleanup
Proper resource management with multiple cleanup hooks:

- Unity lifecycle integration (`OnDestroy`, `OnApplicationQuit`)
- Cancellation tokens for async operations
- Thread join/abort on disconnection
- Lock-protected cleanup sequences

### 4. Error Handling
Comprehensive error handling at multiple levels:

- Protocol validation (UnityProtocol checks message sizes, validates structure)
- JSON parsing errors (JsonParser provides detailed error messages)
- Network exceptions (automatic disconnection and reconnection)
- Callback exceptions (caught and logged without crashing the client)

### 5. Performance Optimization

**Frame Budget Management** (`BidirectionalClientBase.cs:33`):
```csharp
const int MAX_ITEMS_PER_FRAME = 50;
```
Limits response processing to prevent frame drops when many responses arrive simultaneously.

**Pre-allocated Buffers**: The protocol encourages buffer reuse to minimize garbage collection pressure.

**Background Threading**: All blocking I/O happens on background threads, keeping Unity's main thread responsive.

---

## Configuration and Constants

Network configuration is centralized in `Constants.cs` (Core namespace):

```csharp
public static class CommunicationConstants {
    public const string SERVER_HOST = "127.0.0.1";
    public const int STEREO_DETECTION_PORT = 5006;
    public const int LLM_RESULTS_PORT = 5010;
    public const int SEQUENCE_SERVER_PORT = 5013;
    public const int MAX_JSON_LENGTH = 10 * 1024 * 1024;
    public const float RECONNECT_INTERVAL = 2f;
}
```

This allows all clients to share consistent configuration.

---

## Protocol V2 Benefits

The request ID correlation system provides several critical benefits:

1. **Race Condition Prevention**: In multi-robot scenarios, responses can arrive out of order. Request IDs ensure each response is matched to the correct request.

2. **Timeout Handling**: Clients can detect if a response never arrives by tracking pending requests.

3. **Debugging**: Request IDs make it easy to trace requests through logs in both Unity and Python.

4. **Concurrent Operations**: Multiple robots can send requests simultaneously without response mismatches.

---

## Usage Example: Creating a New Client

To create a new bidirectional client, inherit from `BidirectionalClientBase`:

```csharp
using PythonCommunication.Core;

public class MyClient : BidirectionalClientBase<MyResponseType>
{
    protected override string LogPrefix => "[MY_CLIENT]";

    protected override void Awake()
    {
        base.Awake();
        _serverPort = CommunicationConstants.MY_SERVER_PORT;
    }

    protected override MyResponseType ReceiveResponse()
    {
        // 1. Read header
        byte[] headerBuffer = new byte[UnityProtocol.HEADER_SIZE];
        ReadExactly(_stream, headerBuffer, UnityProtocol.HEADER_SIZE);

        // 2. Decode header
        UnityProtocol.DecodeHeader(headerBuffer, 0,
            out MessageType msgType, out uint requestId);

        // 3. Read and parse payload
        string json = ReadJsonString();

        // 4. Parse JSON
        if (JsonParser.TryParse<MyResponseType>(json,
            out var response, out string error))
        {
            return response;
        }

        Debug.LogError($"{LogPrefix} Parse error: {error}");
        return null;
    }

    protected override uint GetResponseRequestId(MyResponseType response)
    {
        // Extract request ID from your response type
        return response.RequestId;
    }

    public void SendMyRequest(string data, Action<MyResponseType> callback = null)
    {
        uint requestId = GenerateRequestId();

        // Encode your message using UnityProtocol
        byte[] message = EncodeMyMessage(data, requestId);

        // Send with callback
        SendRequest(message, requestId, callback);
    }
}
```

---

## Testing

Each component includes comprehensive test coverage:

- Protocol encoding/decoding tests
- Thread safety tests
- Error handling tests
- Integration tests with mock servers

See `ACRLUnity/Assets/Tests/PlayMode/TCPClientTests.cs` for examples.

---

## Summary

The PythonCommunication/Core scripts provide a robust, production-ready foundation for Unity-Python network communication:

- **TCPClientBase**: Connection lifecycle, thread safety, Unity integration
- **BidirectionalClientBase**: Request/response patterns, background threading, callback dispatch
- **UnityProtocol**: Wire protocol V2 implementation with request correlation
- **JsonParser**: Centralized error handling for JSON parsing

Together, these classes eliminate code duplication, provide consistent error handling, and ensure thread-safe operation in Unity's complex lifecycle environment. The Protocol V2 design with request ID correlation enables reliable multi-robot coordination without race conditions.
