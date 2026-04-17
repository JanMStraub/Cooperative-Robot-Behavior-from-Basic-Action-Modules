#!/usr/bin/env python3
"""
send_handoff.py — Test grasp_object_for_handoff directly in Unity without the LLM.

Uses the SequenceServer "EXEC:" prefix to bypass LLM parsing and feed operations
directly into SequenceExecutor. Each step blocks until Unity confirms completion
before the next step begins.

Sequence:
  1. Robot1: grasp_object_for_handoff  (grasps at far end, stays there)
  2. Robot2: receive_handoff            (orients gripper, moves to near end, closes)
  3. Robot1: release_object             (opens gripper — Robot2 now holds it)
  4. Robot1: return_to_start            (return home)

Steps 1 and 2 are dispatched concurrently (matching how Unity sends them)
because grasp_object_for_handoff and receive_handoff synchronize internally
via signal/wait_for_signal.

Usage:
    python tools/send_handoff.py
    python tools/send_handoff.py --object red_bar --grasper Robot1 --receiver Robot2
    python tools/send_handoff.py --grasp-only    # only step 1
    python tools/send_handoff.py --receive-only  # only step 2
    python tools/send_handoff.py --dry-run       # parse without executing
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time

# ── Protocol V2 constants ──────────────────────────────────────────────────────
SEQUENCE_QUERY     = 0x08
RESULT_TYPE        = 0x02
DIRECT_EXEC_PREFIX = "EXEC:"

DEFAULT_HOST     = "127.0.0.1"
DEFAULT_PORT     = 5008
DEFAULT_CAMERA   = "TableStereoCamera"
DEFAULT_OBJECT   = "red_bar"
DEFAULT_GRASPER  = "Robot1"
DEFAULT_RECEIVER = "Robot2"
DEFAULT_TIMEOUT  = 120


# ── Wire format ────────────────────────────────────────────────────────────────

def build_message(ops: list, robot_id: str, camera_id: str, auto_execute: bool, request_id: int) -> bytes:
    """Encode an EXEC: SEQUENCE_QUERY message."""
    cmd_b = (DIRECT_EXEC_PREFIX + json.dumps(ops)).encode("utf-8")
    rob_b = robot_id.encode("utf-8")
    cam_b = camera_id.encode("utf-8")

    msg  = struct.pack("<BI", SEQUENCE_QUERY, request_id)
    msg += struct.pack("<I", len(cmd_b)) + cmd_b
    msg += struct.pack("<I", len(rob_b)) + rob_b
    msg += struct.pack("<I", len(cam_b)) + cam_b
    msg += struct.pack("<B", 1 if auto_execute else 0)
    return msg


def read_response(sock: socket.socket, timeout: float) -> dict:
    """Read a RESULT response (blocks until SequenceServer sends it)."""
    sock.settimeout(timeout)
    header = b""
    while len(header) < 9:
        chunk = sock.recv(9 - len(header))
        if not chunk:
            raise ConnectionError("Server closed connection before sending response")
        header += chunk

    msg_type = header[0]
    resp_len = struct.unpack("<I", header[5:9])[0]

    if msg_type != RESULT_TYPE:
        raise ValueError(f"Unexpected response type 0x{msg_type:02x}")

    body = b""
    while len(body) < resp_len:
        chunk = sock.recv(resp_len - len(body))
        if not chunk:
            raise ConnectionError("Server closed mid-response")
        body += chunk

    return json.loads(body.decode("utf-8"))


def send_ops(
    ops: list,
    robot_id: str,
    label: str,
    host: str,
    port: int,
    camera_id: str,
    timeout: float,
    auto_execute: bool,
    request_id: int,
    out: dict,
):
    """
    Send an operation list to the SequenceServer and store the result in `out`.

    Designed to run in a thread so grasper and receiver can be dispatched
    simultaneously — they synchronise internally via signal/wait_for_signal.
    """
    print(f"  → [{label}] dispatching {len(ops)} op(s) for {robot_id}")
    try:
        msg = build_message(ops, robot_id, camera_id, auto_execute, request_id)
        with socket.create_connection((host, port), timeout=10) as sock:
            sock.sendall(msg)
            out[label] = read_response(sock, timeout + 30)
    except Exception as e:
        out[label] = {"success": False, "error": str(e)}

    success = out[label].get("success", False)
    print(f"    ← [{label}] [{'OK' if success else 'FAILED'}]  {json.dumps(out[label])}")


# ── Handoff sequence ───────────────────────────────────────────────────────────

def run_handoff(
    object_id: str,
    grasper_id: str,
    receiver_id: str,
    host: str,
    port: int,
    camera_id: str,
    timeout: float,
    grasp_only: bool,
    receive_only: bool,
    auto_execute: bool,
) -> bool:
    """
    Dispatch the full handoff sequence.

    Steps 1 & 2 (grasp + receive) run concurrently in threads because
    grasp_object_for_handoff signals when done and receive_handoff waits for
    that signal — they must be in-flight at the same time.

    Steps 3 & 4 (release + home) run sequentially after both complete.
    """
    results: dict = {}
    threads = []

    # ── Steps 1 & 2 (concurrent) ───────────────────────────────────────────────
    if not receive_only:
        grasp_ops = [{
            "operation": "grasp_object_for_handoff",
            "params": {
                "robot_id":           grasper_id,
                "object_id":          object_id,
                "receiving_robot_id": receiver_id,
            },
        }]
        t = threading.Thread(
            target=send_ops,
            args=(grasp_ops, grasper_id, "grasper", host, port,
                  camera_id, timeout, auto_execute, 1, results),
            daemon=True,
        )
        threads.append(t)

    if not grasp_only:
        receive_ops = [{
            "operation": "receive_handoff",
            "params": {
                "robot_id":        receiver_id,
                "object_id":       object_id,
                "source_robot_id": grasper_id,
            },
        }]
        t = threading.Thread(
            target=send_ops,
            args=(receive_ops, receiver_id, "receiver", host, port,
                  camera_id, timeout, auto_execute, 2, results),
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout + 60)

    overall = all(r.get("success", False) for r in results.values())

    # ── Steps 3 & 4: release + home (only if both sides ran and succeeded) ──────
    if not grasp_only and not receive_only and overall and auto_execute:
        print()

        release_ops = [{"operation": "release_object", "params": {"robot_id": grasper_id}}]
        send_ops(release_ops, grasper_id, "release", host, port, camera_id, 20, True, 3, results)

        home_ops = [{"operation": "return_to_start", "params": {"robot_id": grasper_id, "speed": 1.0}}]
        send_ops(home_ops, grasper_id, "home", host, port, camera_id, 60, True, 4, results)

        overall = all(r.get("success", False) for r in results.values())

    return overall


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """Parse arguments and run the handoff."""
    parser = argparse.ArgumentParser(
        description="Test grasp_object_for_handoff in Unity without LLM latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--object",   default=DEFAULT_OBJECT,   help=f"Object ID (default: {DEFAULT_OBJECT})")
    parser.add_argument("--grasper",  default=DEFAULT_GRASPER,  help=f"Grasping robot (default: {DEFAULT_GRASPER})")
    parser.add_argument("--receiver", default=DEFAULT_RECEIVER,  help=f"Receiving robot (default: {DEFAULT_RECEIVER})")
    parser.add_argument("--host",     default=DEFAULT_HOST,      help=f"Host (default: {DEFAULT_HOST})")
    parser.add_argument("--port",     default=DEFAULT_PORT, type=int, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--camera",   default=DEFAULT_CAMERA,   help=f"Camera ID (default: {DEFAULT_CAMERA})")
    parser.add_argument("--timeout",  default=DEFAULT_TIMEOUT, type=float,
                        help=f"Per-step timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--grasp-only",   action="store_true", help="Only run step 1 (grasp)")
    parser.add_argument("--receive-only", action="store_true", help="Only run step 2 (receive)")
    parser.add_argument("--dry-run",  action="store_true", help="Parse without executing")
    parser.add_argument("--pretty",   action="store_true", help="Pretty-print responses")
    args = parser.parse_args()

    if args.grasp_only and args.receive_only:
        print("ERROR: --grasp-only and --receive-only are mutually exclusive")
        sys.exit(1)

    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    print(f"[{mode}] Handoff: {args.grasper} grasps '{args.object}' → {args.receiver} receives")
    print(f"  server={args.host}:{args.port}  camera={args.camera}")
    print()

    try:
        success = run_handoff(
            object_id=args.object,
            grasper_id=args.grasper,
            receiver_id=args.receiver,
            host=args.host,
            port=args.port,
            camera_id=args.camera,
            timeout=args.timeout,
            grasp_only=args.grasp_only,
            receive_only=args.receive_only,
            auto_execute=not args.dry_run,
        )
    except ConnectionRefusedError:
        print(f"\nERROR: Could not connect to {args.host}:{args.port}")
        print("  Start with: python -m orchestrators.RunRobotController")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print()
    print("Done." if success else "Finished with errors.")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
