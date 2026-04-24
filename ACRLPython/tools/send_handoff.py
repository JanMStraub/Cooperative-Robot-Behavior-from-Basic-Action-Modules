#!/usr/bin/env python3
"""
send_handoff.py — Test the explicit handoff pipeline in Unity without the LLM.

Uses the SequenceServer "EXEC:" prefix to bypass LLM parsing and feed operations
directly into SequenceExecutor. Each step blocks until Unity confirms completion
before the next step begins.

Flat step sequence (each is an independent SE operation):
  1. Robot1: grasp_object
  2. Robot1: return_to_start_position  (deterministic joint config)
  3. Robot1: move_to_coordinate        (→ HANDOFF_PRESENTATION_POSITION)
  4. Robot1: adjust_end_effector_orientation(pitch=0,yaw=0,roll=0)  (lock wrist)
  5a. Robot1: signal(r1_at_handoff)
  5b. Robot2: wait_for_signal(r1_at_handoff)   [parallel with 5a]
  6. Robot2: detect_object_stereo      (re-detect at presentation pos)
  7. Robot2: receive_handoff           (orient + move to approach + close gripper)
  8. Robot1: release_object
  9. Robot1: return_to_start_position

Usage:
    python tools/send_handoff.py
    python tools/send_handoff.py --object red_bar --grasper Robot1 --receiver Robot2
    python tools/send_handoff.py --grasp-only    # only step 1
    python tools/send_handoff.py --receive-only  # steps 5-8
    python tools/send_handoff.py --dry-run       # parse without executing
"""

import argparse
import json
import socket
import struct
import sys
import threading

# ── Protocol V2 constants ──────────────────────────────────────────────────────
SEQUENCE_QUERY = 0x08
RESULT_TYPE = 0x02
DIRECT_EXEC_PREFIX = "EXEC:"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5008
DEFAULT_CAMERA = "TableStereoCamera"
DEFAULT_OBJECT = "red_bar"
DEFAULT_GRASPER = "Robot1"
DEFAULT_RECEIVER = "Robot2"
DEFAULT_TIMEOUT = 120

# Handoff presentation position (must match config/Robot.py HANDOFF_PRESENTATION_POSITION)
HANDOFF_X = -0.05
HANDOFF_Y = 0.35
HANDOFF_Z = 0.0


# ── Wire format ────────────────────────────────────────────────────────────────


def build_message(
    ops: list, robot_id: str, camera_id: str, auto_execute: bool, request_id: int
) -> bytes:
    """Encode an EXEC: SEQUENCE_QUERY message."""
    cmd_b = (DIRECT_EXEC_PREFIX + json.dumps(ops)).encode("utf-8")
    rob_b = robot_id.encode("utf-8")
    cam_b = camera_id.encode("utf-8")

    msg = struct.pack("<BI", SEQUENCE_QUERY, request_id)
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
    print(
        f"    ← [{label}] [{'OK' if success else 'FAILED'}]  {json.dumps(out[label])}"
    )



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
    Dispatch the full handoff sequence using explicit flat operations.

    Each step is an independent SE operation with its own Unity ACK.
    Steps 5a/5b (signal/wait) run in parallel threads.
    """
    results: dict = {}
    req = 1  # monotonically increasing request_id

    # ── Step 1: Grasp ─────────────────────────────────────────────────────────
    if not receive_only:
        send_ops(
            [{"operation": "grasp_object", "params": {"robot_id": grasper_id, "object_id": object_id}}],
            grasper_id, "grasp", host, port, camera_id, timeout, auto_execute, req, results,
        )
        req += 1
        if not results.get("grasp", {}).get("success", False):
            return False

    # ── Step 2: Return to start (deterministic joint config) ──────────────────
    if not receive_only:
        print()
        send_ops(
            [{"operation": "return_to_start_position", "params": {"robot_id": grasper_id, "speed": 0.5}}],
            grasper_id, "return_start", host, port, camera_id, 60, auto_execute, req, results,
        )
        req += 1
        if not results.get("return_start", {}).get("success", False):
            return False

    # ── Step 3: Move to handoff presentation position ─────────────────────────
    if not receive_only:
        print()
        send_ops(
            [{"operation": "move_to_coordinate", "params": {
                "robot_id": grasper_id, "x": HANDOFF_X, "y": HANDOFF_Y, "z": HANDOFF_Z,
            }}],
            grasper_id, "move_present", host, port, camera_id, timeout, auto_execute, req, results,
        )
        req += 1
        if not results.get("move_present", {}).get("success", False):
            return False

    # ── Step 4: Lock wrist orientation (deterministic joint 5/6 at presentation pos) ──
    if not receive_only:
        print()
        send_ops(
            [{"operation": "adjust_end_effector_orientation", "params": {
                "robot_id": grasper_id, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            }}],
            grasper_id, "orient_wrist", host, port, camera_id, 30, auto_execute, req, results,
        )
        req += 1
        if not results.get("orient_wrist", {}).get("success", False):
            print("  [WARN] orient_wrist failed — continuing anyway")

    # ── Steps 5a/5b: Signal (R1) + Wait (R2) in parallel ─────────────────────
    if not grasp_only and not receive_only:
        print()
        signal_req = req
        wait_req = req + 1
        req += 2

        signal_out: dict = {}
        wait_out: dict = {}

        t_signal = threading.Thread(target=send_ops, args=(
            [{"operation": "signal", "params": {"robot_id": grasper_id, "event_name": "r1_at_handoff"}}],
            grasper_id, "signal", host, port, camera_id, 30, auto_execute, signal_req, signal_out,
        ))
        t_wait = threading.Thread(target=send_ops, args=(
            [{"operation": "wait_for_signal", "params": {"robot_id": receiver_id, "event_name": "r1_at_handoff"}}],
            receiver_id, "wait_signal", host, port, camera_id, 60, auto_execute, wait_req, wait_out,
        ))

        t_signal.start()
        t_wait.start()
        t_signal.join()
        t_wait.join()

        results.update(signal_out)
        results.update(wait_out)

        if not signal_out.get("signal", {}).get("success", False):
            return False
        if not wait_out.get("wait_signal", {}).get("success", False):
            return False

    # ── Step 6: Receiver detects object at presentation position ──────────────
    if not grasp_only:
        print()
        send_ops(
            [{"operation": "detect_object_stereo", "params": {
                "robot_id": receiver_id, "camera_id": camera_id, "object_color": object_id,
            }}],
            receiver_id, "detect", host, port, camera_id, 60, auto_execute, req, results,
        )
        req += 1
        if not results.get("detect", {}).get("success", False):
            print("  [WARN] detect_object_stereo failed — continuing anyway")

    # ── Step 7: receive_handoff (orient + move to approach + close gripper) ───
    if not grasp_only:
        print()
        send_ops(
            [{"operation": "receive_handoff", "params": {
                "robot_id": receiver_id,
                "object_id": object_id,
                "source_robot_id": grasper_id,
            }}],
            receiver_id, "receive", host, port, camera_id, timeout, auto_execute, req, results,
        )
        req += 1
        if not results.get("receive", {}).get("success", False):
            return False

    # ── Step 8: Grasper releases ──────────────────────────────────────────────
    if not grasp_only and not receive_only:
        print()
        send_ops(
            [{"operation": "release_object", "params": {"robot_id": grasper_id}}],
            grasper_id, "release", host, port, camera_id, 20, auto_execute, req, results,
        )
        req += 1

    # ── Step 9: Grasper returns home ─────────────────────────────────────────
    if not grasp_only and not receive_only:
        print()
        send_ops(
            [{"operation": "return_to_start_position", "params": {"robot_id": grasper_id, "speed": 1.0}}],
            grasper_id, "home", host, port, camera_id, 60, True, req, results,
        )
        req += 1

    return all(r.get("success", False) for r in results.values())


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    """Parse arguments and run the handoff."""
    parser = argparse.ArgumentParser(
        description="Test explicit flat handoff pipeline in Unity without LLM latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--object",
        default=DEFAULT_OBJECT,
        help=f"Object ID (default: {DEFAULT_OBJECT})",
    )
    parser.add_argument(
        "--grasper",
        default=DEFAULT_GRASPER,
        help=f"Grasping robot (default: {DEFAULT_GRASPER})",
    )
    parser.add_argument(
        "--receiver",
        default=DEFAULT_RECEIVER,
        help=f"Receiving robot (default: {DEFAULT_RECEIVER})",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", default=DEFAULT_PORT, type=int, help=f"Port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Camera ID (default: {DEFAULT_CAMERA})",
    )
    parser.add_argument(
        "--timeout",
        default=DEFAULT_TIMEOUT,
        type=float,
        help=f"Per-step timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--grasp-only", action="store_true", help="Only run steps 1-4 (grasp + return + present + orient wrist)"
    )
    parser.add_argument(
        "--receive-only", action="store_true", help="Only run steps 5-8 (detect + approach + close)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse without executing"
    )
    args = parser.parse_args()

    if args.grasp_only and args.receive_only:
        print("ERROR: --grasp-only and --receive-only are mutually exclusive")
        sys.exit(1)

    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    print(
        f"[{mode}] Handoff: {args.grasper} grasps '{args.object}' → {args.receiver} receives"
    )
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
