"""
Review Tool for Generated Operations
=======================================

CLI tool for reviewing, approving, and rejecting dynamically generated operations.

Usage:
    python -m tools.ReviewOperations list              # List pending operations
    python -m tools.ReviewOperations show <id>         # Show operation details
    python -m tools.ReviewOperations approve <id>      # Approve operation
    python -m tools.ReviewOperations reject <id>       # Delete operation
"""

import sys
from pathlib import Path

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from typing import List, Dict, Optional


def _get_generated_dir() -> Path:
    """Get the generated operations directory path."""
    from config.DynamicOperations import GENERATED_OPERATIONS_DIR

    return Path(GENERATED_OPERATIONS_DIR)


def _get_operations() -> List[Dict]:
    """
    List all generated operations with their metadata.

    Returns:
        List of dicts with id, filename, status, command, timestamp
    """
    generated_dir = _get_generated_dir()
    if not generated_dir.exists():
        return []

    operations = []
    for i, py_file in enumerate(sorted(generated_dir.glob("*.py")), 1):
        if py_file.name == "__init__.py":
            continue

        metadata = _parse_metadata(py_file)
        metadata["id"] = i
        metadata["filename"] = py_file.name
        metadata["path"] = str(py_file)
        operations.append(metadata)

    return operations


def _parse_metadata(file_path: Path) -> Dict:
    """
    Parse metadata from a generated operation file header.

    Args:
        file_path: Path to the operation file

    Returns:
        Dict with review_status, generated_at, original_command
    """
    metadata = {
        "review_status": "UNKNOWN",
        "generated_at": "UNKNOWN",
        "original_command": "UNKNOWN",
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# REVIEW_STATUS:"):
                    metadata["review_status"] = line.split(":", 1)[1].strip()
                elif line.startswith("# GENERATED_AT:"):
                    metadata["generated_at"] = line.split(":", 1)[1].strip()
                elif line.startswith("# ORIGINAL_COMMAND:"):
                    metadata["original_command"] = line.split(":", 1)[1].strip()
                elif line and not line.startswith("#"):
                    break
    except Exception:
        pass

    return metadata


def cmd_list(filter_status: Optional[str] = None):
    """
    List generated operations.

    Args:
        filter_status: Optional status filter ("PENDING", "APPROVED")
    """
    operations = _get_operations()

    if filter_status:
        operations = [
            op for op in operations if op["review_status"] == filter_status
        ]

    if not operations:
        print("No generated operations found.")
        return

    print(f"\n{'ID':<4} {'Status':<10} {'Generated':<20} {'Command':<40} {'File'}")
    print("-" * 100)

    for op in operations:
        status = op["review_status"]
        status_display = f"\033[33m{status}\033[0m" if status == "PENDING" else f"\033[32m{status}\033[0m"
        print(
            f"{op['id']:<4} {status:<10} {op['generated_at']:<20} "
            f"{op['original_command'][:40]:<40} {op['filename']}"
        )

    print(f"\nTotal: {len(operations)} operations")


def cmd_show(op_id: int):
    """
    Show the full contents of a generated operation.

    Args:
        op_id: Operation ID (from list command)
    """
    operations = _get_operations()
    op = next((o for o in operations if o["id"] == op_id), None)

    if not op:
        print(f"Operation #{op_id} not found.")
        return

    print(f"\n{'='*60}")
    print(f"Operation #{op['id']}: {op['filename']}")
    print(f"Status: {op['review_status']}")
    print(f"Generated: {op['generated_at']}")
    print(f"Command: {op['original_command']}")
    print(f"{'='*60}\n")

    try:
        content = Path(op["path"]).read_text(encoding="utf-8")
        print(content)
    except Exception as e:
        print(f"Error reading file: {e}")


def cmd_approve(op_id: int):
    """
    Approve a generated operation.

    Args:
        op_id: Operation ID (from list command)
    """
    operations = _get_operations()
    op = next((o for o in operations if o["id"] == op_id), None)

    if not op:
        print(f"Operation #{op_id} not found.")
        return

    from operations.generated import set_review_status

    if set_review_status(op["path"], "APPROVED"):
        print(f"Operation #{op_id} ({op['filename']}) approved.")
        print("Restart the server to activate this operation.")
    else:
        print(f"Failed to approve operation #{op_id}.")


def cmd_reject(op_id: int):
    """
    Reject and delete a generated operation.

    Args:
        op_id: Operation ID (from list command)
    """
    operations = _get_operations()
    op = next((o for o in operations if o["id"] == op_id), None)

    if not op:
        print(f"Operation #{op_id} not found.")
        return

    path = Path(op["path"])
    try:
        path.unlink()
        print(f"Operation #{op_id} ({op['filename']}) rejected and deleted.")
    except Exception as e:
        print(f"Failed to delete operation: {e}")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "list":
        filter_status = sys.argv[2].upper() if len(sys.argv) > 2 else None
        cmd_list(filter_status)

    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: python -m tools.ReviewOperations show <id>")
            return
        cmd_show(int(sys.argv[2]))

    elif command == "approve":
        if len(sys.argv) < 3:
            print("Usage: python -m tools.ReviewOperations approve <id>")
            return
        cmd_approve(int(sys.argv[2]))

    elif command == "reject":
        if len(sys.argv) < 3:
            print("Usage: python -m tools.ReviewOperations reject <id>")
            return
        cmd_reject(int(sys.argv[2]))

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
