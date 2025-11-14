#!/usr/bin/env python3
"""
Example Usage of Robot Operations System
=========================================

This script demonstrates how to use the operations registry to control robots.

Prerequisites:
1. Unity must be running with LLMResultsReceiver active
2. ResultsServer must be running (port 5006)
3. PythonCommandHandler must be attached to a GameObject in Unity

To start the system:
1. Start Unity with your robot scene
2. Run: python -m LLMCommunication.orchestrators.RunAnalyzer
3. Run this script: python -m LLMCommunication.operations.example_usage
"""

from . import get_global_registry, OperationCategory


def main():
    """Demonstrate the operations system"""

    print("=" * 70)
    print("ROBOT OPERATIONS EXAMPLE")
    print("=" * 70)

    # Get the global registry
    registry = get_global_registry()

    # Show registry summary
    print("\n" + registry.generate_summary())

    # Example 1: List all navigation operations
    print("\n" + "=" * 70)
    print("EXAMPLE 1: List Navigation Operations")
    print("=" * 70)

    nav_ops = registry.get_operations_by_category(OperationCategory.NAVIGATION)
    print(f"\nFound {len(nav_ops)} navigation operations:")
    for op in nav_ops:
        print(f"  - {op.name}: {op.description}")

    # Example 2: Get operation details
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Get Operation Details")
    print("=" * 70)

    move_op = registry.get_operation_by_name("move_to_coordinate")
    if move_op:
        print(f"\nOperation: {move_op.name}")
        print(f"ID: {move_op.operation_id}")
        print(f"Category: {move_op.category.value}")
        print(f"Complexity: {move_op.complexity.value}")
        print(f"\nDescription:")
        print(f"  {move_op.description}")
        print(f"\nParameters:")
        for param in move_op.parameters:
            required_str = (
                "Required" if param.required else f"Optional (default: {param.default})"
            )
            range_str = f", range: {param.valid_range}" if param.valid_range else ""
            print(
                f"  - {param.name} ({param.type}): {param.description} [{required_str}{range_str}]"
            )

    # Example 3: Execute operation (requires Unity + ResultsServer running)
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Execute Move Operation")
    print("=" * 70)

    print("\nWARNING: This will send a command to Unity!")
    print("Make sure Unity is running with:")
    print("  1. LLMResultsReceiver active")
    print("  2. ResultsServer running (port 5006)")
    print("  3. PythonCommandHandler attached to scene")
    print("  4. A robot with ID 'Robot1' or 'AR4_Robot'")

    response = input("\nProceed with sending command? (yes/no): ")

    if response.lower() == "yes":
        print("\nExecuting move_to_coordinate operation...")
        result = registry.execute_operation_by_name(
            "move_to_coordinate",
            robot_id="Robot1",  # Change this to match your robot ID
            x=0.2,
            y=0.1,
            z=0.15,
            speed=0.5,
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Result data: {result.result}")
        else:
            print(f"  Error: {result.error}")
    else:
        print("\nSkipping command execution.")

    # Example 4: Export for RAG
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Export for RAG System")
    print("=" * 70)

    response = input("\nExport operations to ./rag_documents? (yes/no): ")

    if response.lower() == "yes":
        registry.export_for_rag("./rag_documents")
        print("\nRAG documents exported! Check ./rag_documents directory.")
        print(
            "These documents can be ingested into a vector database for RAG retrieval."
        )
    else:
        print("\nSkipping RAG export.")

    # Example 5: Show RAG document format
    print("\n" + "=" * 70)
    print("EXAMPLE 5: RAG Document Format")
    print("=" * 70)

    if move_op:
        print("\nSample RAG document (first 500 chars):")
        print("-" * 70)
        rag_doc = move_op.to_rag_document()
        print(rag_doc[:500] + "...")
        print("-" * 70)
        print("\nThis format is optimized for semantic search and LLM retrieval.")

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
