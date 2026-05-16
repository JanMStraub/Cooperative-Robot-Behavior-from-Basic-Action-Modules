#!/usr/bin/env python3
"""AutoRT autonomous task generation orchestrator."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="AutoRT Autonomous Task Generation")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=None,
        help="Robot IDs to use (default: from config)",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Run fully autonomous (no human approval)",
    )
    parser.add_argument(
        "--strategy",
        choices=["balanced", "explore", "exploit", "random"],
        default="balanced",
        help="Task selection strategy",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=None,
        help="Seconds between iterations (default: from config)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to generate per iteration",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    from core.LoggingSetup import setup_logging, enable_file_logging

    setup_logging()
    enable_file_logging()
    if args.verbose:
        import logging as _logging

        _logging.getLogger().setLevel(_logging.DEBUG)

    if args.num_tasks:
        from config import AutoRT as config

        config.MAX_TASK_CANDIDATES = args.num_tasks

    from autort.AutoRTLoop import AutoRTOrchestrator

    orchestrator = AutoRTOrchestrator(
        robot_ids=args.robots,
        autonomous=args.autonomous,
        loop_delay_seconds=args.loop_delay,
        strategy=args.strategy,
    )

    try:
        orchestrator.start()
    except KeyboardInterrupt:
        print("\nAutoRT stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
