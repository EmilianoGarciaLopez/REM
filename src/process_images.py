import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from term_image.exceptions import RenderError
from term_image.image import from_file
from termcolor import colored

from api_models import RobotVision, api_name_to_colloquial, create_model


# -------------------------------------------------------------------------
# Optionally, monkey-patch RobotVision's save_state to do nothing:
def no_op_save_state(self):
    pass


RobotVision.save_state = no_op_save_state
# -------------------------------------------------------------------------


def _parallel_model_inference(model_key, system_prompt, combined_prompt, image_paths):
    """Helper function to run model inference with retries in parallel"""
    MAX_RETRIES = 3
    RETRY_DELAY = 0.15  # seconds

    model_name = api_name_to_colloquial[model_key]
    for attempt in range(MAX_RETRIES):
        try:
            answer = _infer_model(
                model_key, system_prompt, combined_prompt, image_paths
            ).strip()
            logging.info(f"[{model_name}] => {answer}")
            return {model_key: answer}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:  # Last attempt
                logging.error(f"Error from [{model_name}]: {str(e)}")
                return {model_key: f"ERROR: {str(e)}"}
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue


def main():
    """
    A script that:
      1) Recursively finds all trajectory folders containing an "images" subdirectory
      2) For each trajectory:
         a) Gathers all images in the "images" directory
         b) For each known question CSV in that directory, reads each question
         c) Runs every supported model in api_name_to_colloquial on that question in "all_at_once" mode
         d) Appends a column to the CSV (one for each model) with that model's answer

    NOTE: We do NOT save a per-model JSON state. Instead, the model's answers are
    recorded in new CSV columns, adjacent to the question (and any ground truth).
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Process images and record model answers in question CSVs."
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help='Path to the base directory containing trajectory subdirectories (each with an "images" subdirectory).',
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    # Find all trajectory directories containing an "images" subdirectory
    trajectory_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "images" in dirs:
            trajectory_dirs.append(root)

    if not trajectory_dirs:
        logging.error(
            f"No trajectory directories with 'images' subdirectory found in '{base_dir}'"
        )
        sys.exit(1)

    logging.info(f"Found {len(trajectory_dirs)} trajectory directories to process.")

    # Process each trajectory directory
    for trajectory_dir in sorted(trajectory_dirs):
        logging.info(f"\nProcessing trajectory directory: '{trajectory_dir}'")
        image_dir = os.path.join(trajectory_dir, "images")

        # Rest of the processing logic remains the same, just indented under this loop
        # Gather images
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        image_files = [
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.splitext(f)[1].lower() in valid_extensions
        ]
        image_files.sort()
        logging.info(f"Found {len(image_files)} image files to process.")

        # System prompt (no 'size' mentions)
        system_prompt = (
        "You are an agent walking through an environment in Blender. "
        "You will receive a series of images, each taken after taking an action in the environment (either moving straight or turning 15 degrees left/right). "
        "You will also receive a question that you must answer correctly after seeing all images. "
        "You will see objects with a shape and a color. The possible shapes include cuboid, cone, sphere. "
        "The possible colors include red, green, blue, yellow, purple, brown, black, orange. "
        "Please answer the question based on the set of images."
        "Answer as concisely as possible, usually only a single word. If you're asked about a true/false question, "
        "answer with 'yes' or 'no' only. If it's a question where you're asked to compare the number of objects, respond only with whichever object there are more of, or equal, if there are the same number of objects"
        "If you're asked to count objects, answer only with the number (as a number, not in english) of objects you see."
        "If you're asked whether you saw something before, after, or at the same time as another object, "
        "answer only with 'before', 'after', or 'same time' only. If the first time you see an object is in an image before another object, it comes before (and the other comes after). If two objects appear in the same frame together for their first viewing, its same time"
    )

        # Gather annotations if available
        annotations_file = os.path.join(image_dir, "annotations.csv")
        annotations = {}
        if os.path.isfile(annotations_file):
            with open(annotations_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get("image_filename", "").strip()
                    ann = row.get("annotation", "") or ""
                    annotations[fn] = ann.strip()
            logging.info(
                f"Loaded annotations for {len(annotations)} images from '{annotations_file}'."
            )
        else:
            logging.warning(
                "No annotations.csv file found; proceeding without annotations."
            )

        # Build list of (image_path, annotation)
        images_and_annotations = []
        for image_filename in image_files:
            path = os.path.join(image_dir, image_filename)
            annotation = annotations.get(image_filename, "")
            images_and_annotations.append((path, annotation))

        display_images([p for p, _ in images_and_annotations])

        # Known CSV files with questions
        question_csv_candidates = [
            "comparison_questions_in_frame.csv",
            "comparison_questions_out_of_frame.csv",
            "left_right_questions.csv",
            "number_questions.csv",
            "order_preserving_questions.csv",
        ]

        # For each CSV, read questions, then run models sequentially
        for question_csv_name in question_csv_candidates:
            csv_path = os.path.join(image_dir, question_csv_name)
            if not os.path.isfile(csv_path):
                logging.info(f"Question file '{csv_path}' not found. Skipping.")
                continue

            logging.info(f"Processing questions in '{csv_path}' with all known models.")

            # Load rows in memory
            with open(csv_path, "r", newline="", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                if not reader.fieldnames:
                    logging.warning(f"CSV '{csv_path}' has no header row. Skipping.")
                    continue
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # Ensure we have columns for each model
            for model_key, colloquial_name in api_name_to_colloquial.items():
                if colloquial_name not in fieldnames:
                    fieldnames.append(colloquial_name)

            # Process each question...
            for i, row in enumerate(rows):
                question_text = row.get("question", "").strip()
                if not question_text:
                    for model_key, colloquial_name in api_name_to_colloquial.items():
                        row[colloquial_name] = ""
                    continue

                # Prepare question prompt
                combined_prompt, image_paths = _prepare_prompt(
                    question_text, images_and_annotations
                )

                logging.info(f"[Question] {question_text}")
                logging.info(f"[Images] {len(image_paths)} total")
                logging.info("Running each model sequentially for this question...")

                # Run each model sequentially (no concurrency)
                for model_key in api_name_to_colloquial.keys():
                    model_name = api_name_to_colloquial[model_key]
                    logging.info(f"  -> Starting inference for {model_name}")

                    # We wrap the original inference in a try/except to catch all errors
                    try:
                        result = _parallel_model_inference(
                            model_key, system_prompt, combined_prompt, image_paths
                        )
                    except Exception as e:
                        logging.error(f"  -> Error from {model_name}: {e}")
                        result = {model_key: f"ERROR: {str(e)}"}

                    # Store the result in the row
                    for mk, answer in result.items():
                        row[api_name_to_colloquial[mk]] = answer

                # Save progress after each question
                try:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                        logging.info(f"Progress saved to {csv_path}")
                except Exception as e:
                    logging.error(f"Failed to save progress to {csv_path}: {str(e)}")

            logging.info(f"All models processed for '{csv_path}'.")

        logging.info(f"Finished processing trajectory directory: '{trajectory_dir}'")

    logging.info("\nAll trajectory directories processed. Done.")


def _verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.warning(f"Image {path} fails verification: {e}")
        return False


def _prepare_prompt(question_text, images_and_annotations):
    """
    Builds a combined prompt (with annotations), and returns image paths.
    """
    annotations_text = []
    valid_paths = []
    for img_path, ann in images_and_annotations:
        if os.path.exists(img_path) and _verify_image(img_path):
            valid_paths.append(img_path)
            if ann:
                annotations_text.append(ann)
    ann_str = "\n".join([f"{a}" for a in annotations_text])
    combined_prompt = f"{question_text}\n\nAnnotations:\n{ann_str}"
    return combined_prompt, valid_paths


def _infer_model(model_key, system_prompt, combined_prompt, image_paths):
    """
    Calls the model directly without printing from RobotVision, eliminating flicker.
    """
    model = create_model(model_key)
    return model.call_model(
        user_prompt=combined_prompt,
        system_prompt=system_prompt,
        image_paths=image_paths,
    )


def _fetch_answer(robot_vision, question_text, images_and_annotations):
    """Helper for parallel model call."""
    robot_vision.set_goal(question_text)
    robot_vision.process_images(images_and_annotations)
    return robot_vision.get_output()


def display_images(image_paths):
    """Render images directly in the terminal."""
    if not image_paths:
        return
    for i, path in enumerate(image_paths, 1):
        print(colored(f"\nImage {i}:", "magenta"))
        print(colored("=" * 80, "magenta"))
        try:
            img = from_file(path)
            img.width = min(80, os.get_terminal_size().columns)
            print(img)
        except RenderError as e:
            logging.warning(f"Skipping truncated image: {path}, {e}")
            continue
        print(colored("=" * 80, "magenta"))


if __name__ == "__main__":
    main()
