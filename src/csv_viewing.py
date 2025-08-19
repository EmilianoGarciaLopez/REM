#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot left-right question success vs trajectory length for all models."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the aggregated CSV file (e.g., all_questions_aggregated.csv)",
    )
    args = parser.parse_args()

    # Read the aggregated CSV using pandas.
    df = pd.read_csv(args.csv)
    
    # Filter to only include left-right questions.
    df_lr = df[df["question_type"] == "number"].copy()
    
    # Define the list of models.
    models = ["gpt-4o", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
    
    # Set up the matplotlib figure.
    plt.figure(figsize=(10, 6))
    
    # For each model, process its correctness column and plot the success rate vs trajectory length.
    for model in models:
        correctness_col = f"answered_correctly_{model}"
        
        # Filter rows that have either "yes" or "no" in the correctness column.
        df_model = df_lr[df_lr[correctness_col].isin(["yes", "no"])].copy()
        if df_model.empty:
            continue  # Skip if no valid responses.
        
        # Create a new column "success": 1 for "yes", 0 for "no".
        df_model["success"] = df_model[correctness_col].apply(lambda x: 1 if x.lower() == "yes" else 0)
        
        # Group by trajectory length and compute the mean success rate.
        grouped = df_model.groupby("number_of_objects_it_sees")["success"].mean().reset_index()
        grouped = grouped.sort_values("number_of_objects_it_sees")
        
        # Plot the success rate vs trajectory length.
        plt.plot(grouped["number_of_objects_it_sees"], grouped["success"],
                 marker="o", linestyle="-", label=model)
    
    plt.xlabel("Number of Objects")
    plt.ylabel("Success Rate (Fraction Correct)")
    plt.title("Counting Question Success vs Number of Objects (All Models)")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
