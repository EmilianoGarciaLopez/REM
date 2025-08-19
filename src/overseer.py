# overseer.py

import os
import sys
import subprocess
import argparse


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Automate simulations and question answering over multiple trajectories."
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="./src/trajectories",
        help="Path to the directory containing trajectory subdirectories.",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of questions to generate per trajectory per question type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-pro-002",
        choices=["gemini-1.5-pro-002", "gpt-4", "gpt-3.5-turbo", "llama-2-70b-chat"],
        help="Model to use for processing.",
    )
    args = parser.parse_args()

    trajectories_dir = args.trajectories_dir
    num_questions = str(args.num_questions)
    model_name = args.model

    # Ensure the trajectories directory exists
    if not os.path.isdir(trajectories_dir):
        print(f"The trajectories directory '{trajectories_dir}' does not exist.")
        sys.exit(1)

    # Loop over each subdirectory in the trajectories directory
    for dir_name in os.listdir(trajectories_dir):
        trajectory_dir = os.path.join(trajectories_dir, dir_name)
        if os.path.isdir(trajectory_dir):
            print(f"\nProcessing trajectory: {trajectory_dir}")

            # Run the simulation
            simulation_cmd = [
                sys.executable,  # Use the current Python interpreter
                os.path.join(".", "src", "run_simulation.py"),
                "--config-file",
                trajectory_dir,
            ]
            print(f"Running simulation: {' '.join(simulation_cmd)}")
            result = subprocess.run(simulation_cmd)

            if result.returncode != 0:
                print(
                    f"Simulation failed for trajectory '{trajectory_dir}'. Skipping to next trajectory."
                )
                continue

            # Run comparison-question
            comparison_cmd = [
                sys.executable,
                os.path.join(".", "src", "automate_question_answering.py"),
                trajectory_dir,
                "--comparison-question",
                "--num_questions",
                num_questions,
                "--model",
                model_name,
            ]
            print(f"Running comparison questions: {' '.join(comparison_cmd)}")
            subprocess.run(comparison_cmd)

            # Run existence-question
            existence_cmd = [
                sys.executable,
                os.path.join(".", "src", "automate_question_answering.py"),
                trajectory_dir,
                "--existence-question",
                "--num_questions",
                num_questions,
                "--model",
                model_name,
            ]
            print(f"Running existence questions: {' '.join(existence_cmd)}")
            subprocess.run(existence_cmd)

            # Run order-question
            order_cmd = [
                sys.executable,
                os.path.join(".", "src", "automate_question_answering.py"),
                trajectory_dir,
                "--order-question",
                "--num_questions",
                num_questions,
                "--model",
                model_name,
            ]
            print(f"Running order questions: {' '.join(order_cmd)}")
            subprocess.run(order_cmd)

        else:
            print(f"Skipping '{trajectory_dir}', not a directory")


if __name__ == "__main__":
    main()
