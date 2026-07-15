#!/usr/bin/env python3
"""PresentationShowcase.py - live-demo script for the thesis presentation."""

import argparse
import json
import socket
import struct
import sys

SEQUENCE_QUERY = 0x08
RESULT_TYPE = 0x02

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5008
DEFAULT_CAMERA = "TableStereoCamera"
RESPONSE_TIMEOUT_S = 600.0

PARALLEL_PLACE_TASK = (
    "Robot1: detect the blue cube, grasp it, and place it in field A and at the same "
    "time Robot2: detect the green cube, grasp it, and place it in field B."
)

PLACE_BETWEEN_TASK = "Robot1 grasps the blue cube and places it between the red cube and the yellow cube."

HANDOFF_TASK = "Robot1 grasps the red cube and hands it to Robot2."

DUAL_LIFT_TASK = (
    "Robot1 and Robot2 cooperatively handle the red cube. First, Robot1 grasps the "
    "right side of the cube. After Robot1 is in position, Robot2 grasps the cube from "
    "the left side. Once Robot2 has secured the cube, both robots simultaneously lift "
    "it to y=0.15."
)

TASKS = [
    ("Parallel independent pick-and-place", PARALLEL_PLACE_TASK),
    ("Single-robot place-between", PLACE_BETWEEN_TASK),
    ("Dual-robot handoff", HANDOFF_TASK),
    ("Dual-robot lift (current limit)", DUAL_LIFT_TASK),
]


def check_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_sequence_message(
    command_text: str, robot_id: str, camera_id: str, request_id: int
) -> bytes:
    cmd_b = command_text.encode("utf-8")
    rob_b = robot_id.encode("utf-8")
    cam_b = camera_id.encode("utf-8")
    msg = struct.pack("<BI", SEQUENCE_QUERY, request_id)
    msg += struct.pack("<I", len(cmd_b)) + cmd_b
    msg += struct.pack("<I", len(rob_b)) + rob_b
    msg += struct.pack("<I", len(cam_b)) + cam_b
    msg += struct.pack("<B", 1)
    msg += struct.pack("<I", 0)
    return msg


def _read_response(sock: socket.socket, timeout: float) -> dict:
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
            raise ConnectionError("Server closed connection mid-response")
        body += chunk

    return json.loads(body.decode("utf-8"))


def run_task(task_text: str, host: str, port: int, request_id: int) -> dict:
    msg = _build_sequence_message(task_text, "Robot1", DEFAULT_CAMERA, request_id)
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(msg)
        return _read_response(sock, RESPONSE_TIMEOUT_S)


def reset_simulation(host: str, port: int) -> None:
    try:
        command_text = "EXEC:" + json.dumps(
            [{"operation": "reset_simulation", "params": {}}]
        )
        msg = _build_sequence_message(command_text, "system", DEFAULT_CAMERA, 1)
        with socket.create_connection((host, port), timeout=10) as sock:
            sock.sendall(msg)
            _read_response(sock, timeout=15)
    except Exception as e:
        print(f"    (reset_simulation failed, continuing anyway: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Automated live-demo script for the thesis presentation."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"Pre-flight: checking SequenceServer at {args.host}:{args.port} ...")
    if not check_port_open(args.host, args.port):
        print(f"ERROR: SequenceServer not reachable at {args.host}:{args.port}.")
        print("  Start it first: ./start_servers.sh --web 8000")
        sys.exit(1)
    print("  OK")

    for i, (name, task_text) in enumerate(TASKS, start=1):
        print()
        print(f"=== Task {i}/{len(TASKS)}: {name} ===")
        print(f"  task: {task_text}")
        result = run_task(task_text, args.host, args.port, request_id=i)
        status = "OK" if result.get("success") else "FAILED"
        print(f"  -> {status}")

        if i < len(TASKS):
            print("  resetting simulation...")
            reset_simulation(args.host, args.port)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
