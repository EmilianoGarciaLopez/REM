from typing import Dict, List

from openai import OpenAI


def get_active_batches() -> Dict[str, List[str]]:
    """
    Retrieves and categorizes all active batches from OpenAI API.
    Returns a dictionary with counts and details of active batches.
    """
    client = OpenAI()

    # States we consider "active"
    active_states = {"validating", "in_progress", "finalizing"}

    # Initialize results
    results = {
        "total_active": 0,
        "active_batches": [],
        "by_status": {status: [] for status in active_states},
    }

    try:
        # Get all batches
        batches = client.batches.list()

        # Process each batch
        for batch in batches:
            if batch.status in active_states:
                results["total_active"] += 1
                results["active_batches"].append(
                    {
                        "id": batch.id,
                        "status": batch.status,
                        "created_at": batch.created_at,
                        "endpoint": batch.endpoint,
                    }
                )
                results["by_status"][batch.status].append(batch.id)

        return results

    except Exception as e:
        print(f"Error accessing OpenAI API: {str(e)}")
        return None


def main():
    results = get_active_batches()

    if results:
        print(f"\nTotal Active Batches: {results['total_active']}")
        print("\nBreakdown by Status:")
        for status, batches in results["by_status"].items():
            print(f"{status}: {len(batches)} batches")

        if results["active_batches"]:
            print("\nActive Batch Details:")
            for batch in results["active_batches"]:
                print(f"\nBatch ID: {batch['id']}")
                print(f"Status: {batch['status']}")
                print(f"Endpoint: {batch['endpoint']}")
                print(f"Created at: {batch['created_at']}")


if __name__ == "__main__":
    main()
