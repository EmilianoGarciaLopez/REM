import csv
import re
from itertools import combinations
import argparse
import os

# Note remove combinations of form (x,x) where x is the same type of object (red_cube_n, red_cube_n for example) DONE
# Just write out all possible questions, pick a subset
# Reject dataset gens where objects occupy some threshold of screen
# Determine if duplicates by seeing if there is any (red_cube_x, red_cube_y), x != y
# Take set of each object (including suffix), iterate through, see if more than one instance of each (without looking at suffix)
# Make duplicates dictionary, which maps object without suffix to boolean of whether duplicate


def setup_argparser():
    parser = argparse.ArgumentParser(
        description="Process annotations from a trajectory run directory"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to trajectory run directory (e.g., ./trajectories/0007-run/)",
    )
    return parser


# Data Structures Needed For Each type of question:
# Set-based Memory
#
# More x than y, How many of x, was there an x, list all unique objects
#   First, identify duplicates, as you want to ask about duplicates else random
#   Remove integer suffixes, create dictionary mapping objects to number
#   Then separate into cooccur/non-cooccur dictionaries (mapping to number), attempt to ask questions about each set
#
# Order Preserving Memory
#   List objects in order they appear (given all objects), x or y first or same time (same frame is arbitrary)
#   Get dictionary mapping object to frame number of first appearance
# Directional Memory
#   Was object to left or right of other object, both inside image and up to 3 images apart


def parse_annotations(csv_path):
    # Dictionary that will map image_number (int) -> list of (object_name, percentage) tuples
    image_data = {}

    # Open the CSV file
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_filename = row["image_filename"]
            visible_objects_str = row["visible_objects"]

            # Extract the image number from the filename.
            # The filename format is "image_000X_something.png"
            # We'll parse the "000X" substring and convert to int.
            # For example: image_0001_move_forward_1m.png -> image_number = 1
            match = re.search(r"image_(\d+)", image_filename)
            if not match:
                # If no match, skip this line (or raise an error)
                continue
            image_number = int(match.group(1))

            # The visible_objects field is a comma-separated list of objects in the format:
            # "Name (XX.XX%), OtherName (YY.YY%), ..."
            # We can split by commas, but we must be careful that object names do not contain commas.
            # In your example, each object entry looks like: objectName (XX.XX%)
            # Since each object is well-formed and separated by commas, we can directly split by ", "
            # after trimming extra whitespace.

            objects_list = visible_objects_str.split(",")
            # After splitting by ',', some objects might have leading/trailing spaces.
            objects_list = [obj.strip() for obj in objects_list if obj.strip()]

            # Now we have a list like ["GroundPlane (27.86%)", "large_red_metallic_cylinder_2 (17.47%)", ...]
            # We need to extract the object name and the percentage.

            parsed_objects = []
            for obj_str in objects_list:
                # The pattern is something like: "Name (XX.XX%)"
                # We can split on the last '(' to isolate name from percentage.
                # Alternatively, we can use a regex.

                # Regex approach:
                # Pattern: (.*)\s+\(([\d.]+)%\)$
                # Group 1: object name
                # Group 2: percentage
                m = re.match(r"^(.*)\s+\(([\d.]+)%\)$", obj_str)
                if m:
                    name = m.group(1).strip()
                    percentage_str = m.group(2).strip()
                    percentage = float(percentage_str)

                    # Skip the ground plane
                    if name == "GroundPlane":
                        continue

                    parsed_objects.append((name, percentage))
                else:
                    # If it doesn't match, we can skip or raise an error.
                    # But ideally all lines match this pattern.
                    continue

            image_data[image_number] = parsed_objects

    return image_data


def get_object_type(obj_name):
    # Remove trailing underscore + number pattern (e.g., "_17") if it exists.
    return re.sub(r"_\d+$", "", obj_name)


def generate_pairs_with_and_without_cooccurrences(image_data):
    # Map each object to the set of images in which it appears
    object_to_images = {}
    for img_num, objects in image_data.items():
        for obj_name, pct in objects:
            if obj_name not in object_to_images:
                object_to_images[obj_name] = set()
            object_to_images[obj_name].add(img_num)

    # Group objects by base type
    type_to_objects = {}
    for obj_name in object_to_images:
        base_type = get_object_type(obj_name)
        if base_type not in type_to_objects:
            type_to_objects[base_type] = []
        type_to_objects[base_type].append(obj_name)

    # duplicates: base_type -> boolean (True if multiple variants)
    duplicates = {
        base_type: (len(objs) > 1) for base_type, objs in type_to_objects.items()
    }
    # object_counts: base_type -> number of unique instances
    object_counts = {
        base_type: len(objs) for base_type, objs in type_to_objects.items()
    }
    # All unique base object types
    unique_base_objects = list(type_to_objects.keys())

    all_objects = list(object_to_images.keys())
    object_pairs = combinations(all_objects, 2)

    co_occurrences = set()
    non_co_occurrences = set()

    for obj1, obj2 in object_pairs:
        # Skip if they are the same type
        if get_object_type(obj1) == get_object_type(obj2):
            continue

        imgs_obj1 = object_to_images[obj1]
        imgs_obj2 = object_to_images[obj2]

        # Find intersection to identify true co-occurrences (same image)
        common_imgs = imgs_obj1.intersection(imgs_obj2)

        if common_imgs:
            # They appear together in at least one image
            for ci in common_imgs:
                co_occurrences.add((obj1, obj2, ci, ci))
        else:
            # They never co-occur in the same image
            for i1 in imgs_obj1:
                for i2 in imgs_obj2:
                    non_co_occurrences.add((obj1, obj2, i1, i2))

    # Determine which base-type pairs co-occur
    co_occurring_base_pairs = set()
    for o1, o2, _, _ in co_occurrences:
        bt1 = get_object_type(o1)
        bt2 = get_object_type(o2)
        # Normalize order so that (A,B) is the same as (B,A)
        if bt1 > bt2:
            bt1, bt2 = bt2, bt1
        co_occurring_base_pairs.add((bt1, bt2))

    # Now find all possible pairs of unique base objects
    base_pairs = set()
    for bt1, bt2 in combinations(unique_base_objects, 2):
        # Only consider pairs of distinct base types
        if bt1 != bt2:
            # Sort them so that pairs are always in alphabetical order
            if bt1 > bt2:
                bt1, bt2 = bt2, bt1
            base_pairs.add((bt1, bt2))

    # never_co_occurring_base_pairs are those that are not in co_occurring_base_pairs
    never_co_occurring_base_pairs = base_pairs - co_occurring_base_pairs

    return (
        co_occurrences,
        non_co_occurrences,
        duplicates,
        object_counts,
        unique_base_objects,
        never_co_occurring_base_pairs,
    )


def find_earliest_appearances(image_data):
    """
    Finds the earliest appearance of each unique object type in the image data.

    Args:
        image_data: A dictionary mapping image numbers to a list of (object_name, percentage) tuples.

    Returns:
        A list of sets, where each set contains the image numbers representing the earliest appearance of an object type.
    """

    # Dictionary to store the earliest image number for each object type
    earliest_appearance = {}

    for img_num, objects in image_data.items():
        for obj_name, _ in objects:
            obj_type = get_object_type(obj_name)

            if obj_type not in earliest_appearance:
                earliest_appearance[obj_type] = img_num
            else:
                earliest_appearance[obj_type] = min(
                    earliest_appearance[obj_type], img_num
                )

    # Group image numbers by object types that appear at the same time
    appearance_groups = {}
    for obj_type, img_num in earliest_appearance.items():
        if img_num not in appearance_groups:
            appearance_groups[img_num] = set()
        appearance_groups[img_num].add(obj_type)

    # Convert the dictionary to a list of sets
    result = list(appearance_groups.values())

    return result


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    # Construct path to annotations.csv
    csv_file_path = os.path.join(args.path, "images", "annotations.csv")

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Could not find annotations.csv at {csv_file_path}")

    image_dict = parse_annotations(csv_file_path)
    earliest_appearances = find_earliest_appearances(image_dict)
    print(earliest_appearances)
