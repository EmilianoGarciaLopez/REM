import sys

import openai
from openai import OpenAI


def yield_all_batches(client, limit=50):
    """
    Yields all batches from OpenAI in multiple pages, if needed.
    """
    page = client.batches.list(limit=limit)
    while True:
        # Yield every batch in the current page
        for batch in page:
            yield batch

        # If no next page, break
        if not page.has_next_page:
            break

        # Attempt to get the next page
        try:
            page = page.get_next_page()
        except RuntimeError as e:
            # Work around the "No next page expected" error
            print(f"Warning: {e}")
            break


def cancel_nonfailed_batches(client, limit=50):
    """
    Cancels all *non-failed* (and not-yet-final) batches from the OpenAI client.
    """
    for batch in yield_all_batches(client, limit=limit):
        batch_id = batch.id
        batch_status = batch.status.lower()

        # Skip batches that are already done (failed, cancelled, or completed)
        if batch_status in ["failed", "cancelled", "completed"]:
            # print(f"Skipping batch {batch_id} because it is '{batch_status}'.")
            continue

        # Otherwise, try cancelling
        print(f"Cancelling batch {batch_id} with status '{batch_status}'...")
        try:
            client.batches.cancel(batch_id)
        except openai.ConflictError as e:
            # e.g. "Cannot cancel a batch with status 'failed'."
            print(f"Cannot cancel batch {batch_id}: {e}")
        except Exception as e:
            print(f"Error cancelling batch {batch_id}: {e}")
            # If you prefer to halt on any error, uncomment:
            # sys.exit(1)


if __name__ == "__main__":
    client = OpenAI()
    cancel_nonfailed_batches(client)
    print("Finished attempting to cancel all non-failed, non-cancelled batches.")
