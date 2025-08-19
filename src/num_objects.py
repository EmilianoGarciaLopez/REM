#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot a histogram of the total number of objects in a scene (using the 'number_of_objects_in_scene' column)."
    )
    parser.add_argument("--csv", default="src/all_questions_aggregated.csv",
                        help="Path to the CSV file (default: %(default)s)")
    args = parser.parse_args()
    
    # Read the CSV file.
    df = pd.read_csv(args.csv)
    
    # Convert the column to numeric (coercing errors to NaN) and drop rows with missing values.
    df["number_of_objects_in_scene"] = pd.to_numeric(df["number_of_duplicates"], errors="coerce")
    df = df.dropna(subset=["number_of_duplicates"])
    
    # Convert the values to integers (if desired).
    df["number_of_duplicates"] = df["number_of_duplicates"].astype(int)
    
    # Create the histogram.
    plt.figure(figsize=(8, 6))
    plt.hist(df["number_of_duplicates"], bins='auto', color='skyblue', edgecolor='black')
    plt.xlabel("Total Number of Objects in Scene")
    plt.ylabel("Frequency")
    plt.title("Histogram of Total Number of Objects in Scene")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
