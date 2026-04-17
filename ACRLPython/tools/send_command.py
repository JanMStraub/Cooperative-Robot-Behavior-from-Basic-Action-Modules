#!/usr/bin/env python3
"""
send_command.py — Send commands directly to Unity without going through the LLM.

Connects to the SequenceServer (port 5008) and sends pre-parsed operations using
the "EXEC:<json>" prefix, which bypasses CommandParser/LLM entirely and feeds the
command list straight into SequenceExecutor → Unity. Response and completion
tracking work exactly as they do for normal Unity-initiated commands.

Usage:
    python tools/send_command.py move   --robot Robot1 --x 0.3 --y 0.2 --z 0.1
    python tools/send_command.py grasp  --robot Robot1 --object red_cube
    python tools/send_command.py gripper --robot Robot1 --open
    python tools/send_command.py gripper --robot Robot1 --close
    python tools/send_command.py release --robot Robot1
    python tools/send_command.py home   --robot Robot1
    python tools/send_command.py raw    --ops '[{"operation":"grasp_object","params":{"robot_id":"Robot1","object_id":"red_cube"}}]'

Options:
    --host HOST     SequenceServer host (default: 127.0.0.1)
    --port PORT     SequenceServer port (default: 5008)
    --timeout SEC   Per-operation timeout in seconds (default: 120)
    --dry-run       Parse without executing (returns plan, no Unity movement)
    --pretty        Pretty-print JSON response
"""

import argparse
import json
import socket
import struct
import sys
import time

# ── Protocol V2 constants (mirrors core/UnityProtocol.py) ─────────────────────
SEQUENCE_QUERY = 0x08   # MessageType.SEQUENCE_QUERY
RESULT_TYPE    = 0x02   # MessageType.RESULT

DEFAULT_HOST    = "127.0.0.1"
DEFAULT_PORT    = 5008
DEFAULT_ROBOT   = "Robot1"
DEFAULT_CAMERA  = "TableStereoCamera"
DEFAULT_TIMEOUT = 120   # grasp + trajectory can take 60–90 s

DIRECT_EXEC_PREFIX = "EXEC:"


# ── Wire format ────────────────────────────────────────────────────────────────

def build_sequence_message(
    command_text: str,
    robot_id: str,
    camera_id: str,
    auto_execute: bool,
    request_id: int,
) -> bytes:
    """
    Encode a SEQUENCE_QUERY message in Protocol V2 wire format.

    Format: [type:1][request_id:4LE][cmd_len:4LE][cmd:N][robot_id_len:4LE][robot_id:N][cam_len:4LE][cam:N][auto_execute:1]
    """
    cmd_b = command_text.encode("utf-8")
    rob_b = robot_id.encode("utf-8")
    cam_b = camera_id.encode("utf-8")

    msg  = struct.pack("<BI", SEQUENCE_QUERY, request_id)
    msg += struct.pack("<I", len(cmd_b)) + cmd_b
    msg += struct.pack("<I", len(rob_b)) + rob_b
    msg += struct.pack("<I", len(cam_b)) + cam_b
    msg += struct.pack("<B", 1 if auto_execute else 0)
    return msg


def read_response(sock: socket.socket, timeout: float) -> dict:
    """
    Read the RESULT response from the SequenceServer.

    Format: [type:1][request_id:4LE][response_len:4LE][json:N]
    The SequenceServer sends this after the entire operation sequence completes
    (or fails), so this may block for the full operation duration.
    """
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
        raise ValueError(
            f"Unexpected response type 0x{msg_type:02x} (expected 0x{RESULT_TYPE:02x})"
        )

    body = b""
    while len(body) < resp_len:
        chunk = sock.recv(resp_len - len(body))
        if not chunk:
            raise ConnectionError("Server closed connection mid-response")
        body += chunk

    return json.loads(body.decode("utf-8"))


def send(
    ops: list,
    robot_id: str,
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
    camera_id: str = DEFAULT_CAMERA,
    request_id: int = 1,
) -> dict:
    """
    Send a pre-parsed operation list to the SequenceServer and return the result.

    The command text is prefixed with "EXEC:" so SequenceQueryHandler skips the
    LLM and passes the list directly to SequenceExecutor.
    """
    command_text = DIRECT_EXEC_PREFIX + json.dumps(ops)
    auto_execute = not dry_run
    msg = build_sequence_message(command_text, robot_id, camera_id, auto_execute, request_id)

    # Generous socket timeout: the response only arrives after Unity finishes all ops
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(msg)
        return read_response(sock, timeout + 30)  # +30s headroom over per-op timeout


# ── Operation builders ─────────────────────────────────────────────────────────

def ops_move(args) -> list:
    """Build a move_to_coordinate operation."""
    return [{
        "operation": "move_to_coordinate",
        "params": {
            "robot_id": args.robot,
            "x": args.x,
            "y": args.y,
            "z": args.z,
            "speed": getattr(args, "speed", 1.0),
            "use_advanced_planning": not getattr(args, "no_advanced", False),
        },
    }]


def ops_grasp(args) -> list:
    """Build a grasp_object operation."""
    return [{
        "operation": "grasp_object",
        "params": {
            "robot_id":              args.robot,
            "object_id":             args.object,
            "preferred_approach":    args.approach,
            "pre_grasp_distance":    args.pre_grasp_distance,
            "enable_retreat":        not args.no_retreat,
            "retreat_distance":      args.retreat_distance,
            "use_advanced_planning": not args.no_advanced,
        },
    }]


def ops_gripper(args) -> list:
    """Build a control_gripper operation."""
    return [{
        "operation": "control_gripper",
        "params": {
            "robot_id":     args.robot,
            "open_gripper": args.open,
        },
    }]


def ops_release(args) -> list:
    """Build a release_object operation."""
    return [{"operation": "release_object", "params": {"robot_id": args.robot}}]


def ops_home(args) -> list:
    """Build a return_to_start operation."""
    return [{
        "operation": "return_to_start",
        "params": {
            "robot_id": args.robot,
            "speed":    getattr(args, "speed", 1.0),
        },
    }]


def ops_raw(args) -> list:
    """Parse a raw JSON ops list from the CLI."""
    ops = json.loads(args.ops)
    if not isinstance(ops, list):
        raise ValueError("--ops must be a JSON array of operation objects")
    return ops


# ── CLI ────────────────────────────────────────────────────────────────────────

def add_common(p: argparse.ArgumentParser):
    """Attach shared arguments to a sub-command parser."""
    p.add_argument("--robot",   default=DEFAULT_ROBOT,  help=f"Robot ID (default: {DEFAULT_ROBOT})")
    p.add_argument("--host",    default=DEFAULT_HOST,   help=f"Server host (default: {DEFAULT_HOST})")
    p.add_argument("--port",    default=DEFAULT_PORT, type=int, help=f"Port (default: {DEFAULT_PORT})")
    p.add_argument("--timeout", default=DEFAULT_TIMEOUT, type=float,
                   help=f"Per-operation timeout in seconds (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--camera",  default=DEFAULT_CAMERA, help=f"Camera ID (default: {DEFAULT_CAMERA})")
    p.add_argument("--dry-run", action="store_true",    help="Parse without executing (no Unity movement)")
    p.add_argument("--pretty",  action="store_true",    help="Pretty-print JSON response")


def main():
    """Parse CLI and dispatch."""
    parser = argparse.ArgumentParser(
        description="Send operations directly to Unity via SequenceServer — no LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    # move
    p = sub.add_parser("move", help="Move end-effector to a world coordinate")
    add_common(p)
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-advanced", action="store_true")
    p.set_defaults(builder=ops_move)

    # grasp
    p = sub.add_parser("grasp", help="Grasp an object by ID")
    add_common(p)
    p.add_argument("--object", required=True, help="Object ID (e.g. red_cube)")
    p.add_argument("--approach", default="top", choices=["top", "front", "side"])
    p.add_argument("--pre-grasp-distance", type=float, default=0.15, dest="pre_grasp_distance")
    p.add_argument("--no-retreat",  action="store_true")
    p.add_argument("--retreat-distance", type=float, default=0.1, dest="retreat_distance")
    p.add_argument("--no-advanced", action="store_true")
    p.set_defaults(builder=ops_grasp)

    # gripper
    p = sub.add_parser("gripper", help="Open or close the gripper")
    add_common(p)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--open",  dest="open", action="store_true",  help="Open gripper")
    grp.add_argument("--close", dest="open", action="store_false", help="Close gripper")
    p.set_defaults(builder=ops_gripper)

    # release
    p = sub.add_parser("release", help="Release held object (open gripper)")
    add_common(p)
    p.set_defaults(builder=ops_release)

    # home
    p = sub.add_parser("home", help="Return robot to start position")
    add_common(p)
    p.add_argument("--speed", type=float, default=1.0)
    p.set_defaults(builder=ops_home)

    # raw
    p = sub.add_parser("raw", help="Send a raw JSON operation list")
    add_common(p)
    p.add_argument("--ops", required=True,
                   help='JSON array, e.g. \'[{"operation":"grasp_object","params":{"robot_id":"Robot1","object_id":"red_cube"}}]\'')
    p.set_defaults(builder=ops_raw)

    args = parser.parse_args()

    try:
        ops = args.builder(args)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR building operation list: {e}")
        sys.exit(1)

    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    print(f"→ [{mode}] {args.cmd.upper()}  robot={args.robot}  server={args.host}:{args.port}")
    print(f"  ops: {json.dumps(ops)}")
    print()

    try:
        result = send(
            ops=ops,
            robot_id=args.robot,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            dry_run=args.dry_run,
            camera_id=args.camera,
        )
    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to {args.host}:{args.port}")
        print("  Is the server running?  python -m orchestrators.RunRobotController")
        sys.exit(1)
    except (TimeoutError, socket.timeout):
        print(f"ERROR: Timed out after {args.timeout}s waiting for response")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    success = result.get("success", False)
    print(f"← [{'OK' if success else 'FAILED'}]")
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
