import argparse
import csv
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# NEW: Import from the new SDK.
from google import genai

# Introduce a separate variable for the real model name (no '-TEXT' suffix)
MODEL_NAME_REAL = "gemini-1.5-flash-latest"
MODEL_COLUMN_PRO_TEXT = "gemini-1.5-flash-latest-TEXT-NEW-PROMPT"

SYSTEM_PROMPT_TEXT = (
    "You are an agent walking through an environment in Blender. "
    "Instead of seeing images, you receive textual descriptions of all objects visible in each step. "
    "Each step has lines of text in the format 'visible_objects' describing the objects, which include shapes "
    "(cuboid, cone, sphere) and colors (red, green, blue, yellow, purple, brown, black, orange). "
<<<<<<< HEAD
    "The visible objects are in left to right order textually, and the first object is the one furthest to the left. Each unique object has a unique ID (red_cone_1, for example), and may persist across frames. It is the same object. "
    "Use these textual descriptions to correctly answer the final question as concisely as possible.\n"
    "Answer with the same constraints: if the question is a true/false question, answer with 'yes' or 'no'. "
    "If it's a comparison question (which object is more frequent), respond with just that object or 'equal'. "
    "If it's about counting, respond with the number of objects as a digit (not word). If it's about order or timing (what you saw first), "
    "respond only with 'before', 'after', or 'same time'."
=======
    "The visible objects are in left to right order textually, and the first object is the one furthest to the left. "
    "Use these textual descriptions to carefully reason about the final question. Provide your reasoning before providing your final answer."
    "\n\n"
    "IMPORTANT: Provide your FINAL answer **only** in triple-backtick code block format, for example:\n"
    "```YES```\n"
    "Answer with the same constraints: if the question is a true/false question, fill in your final answer with 'yes' or 'no'. "
    "If it's a comparison question (which object is more frequent), fill in your final answer with just that object or 'equal'. "
    "For example, if the question is 'Are there more spheres or green objects?', your final can be 'green objects'. "
    "If it's about counting, fill in your final answer with the number of objects as a digit (not word). "
    "If it's about order or timing (what you saw first), fill in your final answer with 'before', 'after', or 'same time'."
>>>>>>> 6d563a635f4080589ebcde0119da8f58e3c75536
)

# The CSVs that might contain questions
QUESTION_CSVS = [
    "comparison_questions_in_frame.csv",
    "comparison_questions_out_of_frame.csv",
    "left_right_questions.csv",
    "number_questions.csv",
    "order_preserving_questions.csv",
]


def get_custom_id(run_dir: str, csv_name: str, row_index: int, model_name: str) -> str:
    """
    Sanitize the run_dir, CSV filename, and model name then append __row_{row_index}.
    This ensures uniqueness across multiple run directories and model usage.
    """
    safe_run_dir = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(run_dir))[:50]
    safe_csv_name = re.sub(r"[^a-zA-Z0-9_-]", "_", csv_name)[:50]
    safe_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:50]
    result = f"{safe_run_dir}__{safe_csv_name}__{safe_model_name}__row_{row_index}"
    return result[:64]


def parse_final_answer(raw_text: str) -> str:
    """
    Look for the text enclosed in triple backticks. If found, return whatever is inside
    the last triple backtick block. Otherwise return the entire string stripped.
    """
    matches = re.findall(r"```(.*?)```", raw_text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return raw_text.strip()


def call_gemini_api_text(
    model_name,
    env_description_text,
    question,
    extra_logging=False,
    max_retries=4,
    initial_delay=3,
):
    """
    Calls the specified Gemini model with textual environment descriptions (no images).
    Returns the response text or an error message.
    Implements exponential backoff retry logic with reset on success.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Build the list of content parts: system prompt, environment description, and the user's question.
            content_parts = [
                SYSTEM_PROMPT_TEXT,
                env_description_text,
                question,
            ]

            if extra_logging:
                logging.info(f"Request for model={model_name} question={question}")

            # NEW: Call the new client API. The new SDK uses keyword arguments.
            response = client.models.generate_content(
                model=model_name, contents=content_parts
            )

            if attempt > 0:
                logging.info(f"Successfully recovered after {attempt + 1} attempts")
            return response.text.strip()

        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                logging.warning(
                    f"Error on attempt {attempt + 1}/{max_retries} for model={model_name}: {exc}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    logging.error(
        f"Final error calling Gemini API for model={model_name}: {last_exception}"
    )
    return f"ERROR: {last_exception}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Parallel text-based questions to Gemini for visible objects, logging debug info."
    )
    parser.add_argument(
        "--base_dir",
        default="src/Fixed_Blender_Questions_Generated_And_Answered/duplicates_7",
        help="Base directory with subfolders (e.g., 0001-run). Each should have an images/ subdir.",
    )
    parser.add_argument(
        "--logging",
        default="false",
        help="Set to 'true' to enable extra logging of question and messages.",
    )

    args = parser.parse_args()
    extra_logging = args.logging.strip().lower() == "true"

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        logging.error(f"{base_dir} does not exist.")
        sys.exit(1)

    # Gather all run directories
    run_dirs = sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    )

    # We'll store an in-memory map of run_dir -> csv_name -> row data for final writing
    csv_data_tracker = {}

    # Helper to load environment text from images/annotations.csv.
    # This function combines the "annotation" and "visible_objects" columns (if present).
    def load_environment_text(images_dir):
        annotations_file = os.path.join(images_dir, "annotations.csv")
        lines = []
        if os.path.isfile(annotations_file):
            logging.info(f"Found annotations.csv in {images_dir}, loading text data...")
            with open(annotations_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ann_text = row.get("annotation", "").strip()
                    vis_text = row.get("visible_objects", "").strip()
                    if ann_text:
                        lines.append(f"Annotation: {ann_text}")
                    if vis_text:
                        lines.append(f"Visible Objects: {vis_text}")
        else:
            logging.info(f"No annotations.csv in {images_dir}, skipping.")
            return "Annotations:\n(None)"

        if not lines:
            return "Annotations:\n(None)"
        else:
            return "Environment Text:\n" + "\n".join(lines)

    # Build a list of requests (text-only; no images)
    all_requests = []

    for run_dir in run_dirs:
        images_dir = os.path.join(run_dir, "images")
        if not os.path.isdir(images_dir):
            logging.info(f"No images/ folder found in {run_dir}, skipping.")
            continue

        logging.info(f"\nGathering data from run directory: {run_dir}")

        env_text_block = load_environment_text(images_dir)

        # Prepare an entry in csv_data_tracker for this run_dir
        csv_data_tracker[run_dir] = {}

        for csv_name in QUESTION_CSVS:
            csv_path = os.path.join(images_dir, csv_name)
            if not os.path.isfile(csv_path):
                continue

            with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
                reader = csv.DictReader(f_in)
                if not reader.fieldnames:
                    continue
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # Ensure we have the new TEXT column
            if MODEL_COLUMN_PRO_TEXT not in fieldnames:
                fieldnames.append(MODEL_COLUMN_PRO_TEXT)

            csv_data_tracker[run_dir][csv_name] = {
                "fieldnames": fieldnames,
                "rows": rows,
            }

            for i, row in enumerate(rows):
                question = row.get("question", "").strip()
                if not question:
                    continue

                # If the output cell is empty or has an error, we queue it up.
                existing_text_version = row.get(MODEL_COLUMN_PRO_TEXT, "")
                if not existing_text_version or existing_text_version.startswith(
                    "ERROR:"
                ):
                    all_requests.append(
                        {
                            "run_dir": run_dir,
                            "csv_name": csv_name,
                            "row_index": i,
                            "model_name": MODEL_NAME_REAL,
                            "question": question,
                            "env_text_block": env_text_block,
                        }
                    )

    if not all_requests:
        logging.info("No new requests to process. Exiting.")
        return

    max_workers = 12
    logging.info(
        f"Processing {len(all_requests)} requests (text-based) in parallel, up to {max_workers} at a time..."
    )
    results_map = {}  # (run_dir, csv_name, row_index, model_name) -> answer

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {}
        for req in all_requests:
            future = executor.submit(
                call_gemini_api_text,
                req["model_name"],
                req["env_text_block"],
                req["question"],
                extra_logging,
            )
            future_to_request[future] = req

        for future in as_completed(future_to_request):
            req = future_to_request[future]
            try:
                # Receive the raw response from Gemini
                raw_answer = future.result()
                # Parse out the text inside triple backticks for the final answer
                final_answer = parse_final_answer(raw_answer)
            except Exception as exc:
                logging.error(f"Request failed: {exc}")
                final_answer = f"ERROR: {exc}"

            run_dir = req["run_dir"]
            csv_name = req["csv_name"]
            row_index = req["row_index"]
            model_name = req["model_name"]

            # Save final_answer in results_map rather than raw_answer
            results_map[(run_dir, csv_name, row_index, model_name)] = final_answer

    # Update CSV files with the results
    for run_dir, csv_dict in csv_data_tracker.items():
        for csv_name, data_obj in csv_dict.items():
            fieldnames = data_obj["fieldnames"]
            rows = data_obj["rows"]

            for i, row in enumerate(rows):
                key_text = (run_dir, csv_name, i, MODEL_NAME_REAL)
                if key_text in results_map:
                    row[MODEL_COLUMN_PRO_TEXT] = results_map[key_text]

            csv_path = os.path.join(run_dir, "images", csv_name)
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logging.info(f"Updated {csv_path} with TEXT-based Gemini results.")

    logging.info("All text-based requests completed and CSVs updated. Done.")


if __name__ == "__main__":
    # Load environment variables from .env if present (optional)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment. Set it in your .env file.")
    client = genai.Client(api_key=api_key)
    main()
