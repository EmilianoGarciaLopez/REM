import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_number_questions(csv_path):
    """
    Analyze number questions from the complete CSV file
    and generate visualizations comparing model performance
    """
    # Load the full CSV file
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract model names for consistency
    models = [
        "gpt-4o",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest", 
        "nova-lite-v1",
        "llama-3.2-11b-vision-instruct",
        "gemini-1.5-flash-latest-TEXT-NEW-PROMPT"
    ]
    
    # Filter for number questions only
    number_questions = df[df['question_type'] == 'number'].copy()
    print(f"Found {len(number_questions)} number questions in the dataset.")
    
    # Convert ground truth and model columns to numeric
    for col in ['ground_truth', 'max_number_in_single_frame'] + models:
        number_questions[col] = pd.to_numeric(number_questions[col], errors='coerce')
    
    # Generate visualizations
    plot_ground_truth_vs_model_answers(number_questions, models)
    plot_max_in_frame_ratio(number_questions, models)
    
    # Return the processed data frame for any additional analysis
    return number_questions

def plot_ground_truth_vs_model_answers(df, models):
    """
    Create simple mean lines of ground truth vs model answers
    """
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Define colors for each model
    colors = {
        "gpt-4o": "#4285F4",
        "gemini-1.5-pro-latest": "#EA4335",
        "gemini-1.5-flash-latest": "#FBBC05", 
        "nova-lite-v1": "#34A853",
        "llama-3.2-11b-vision-instruct": "#8B4513",
        "gemini-1.5-flash-latest-TEXT-NEW-PROMPT": "#FF6D01"
    }
    
    # Define short names for models
    short_names = {
        "gpt-4o": "GPT-4o",
        "gemini-1.5-pro-latest": "Gemini Pro",
        "gemini-1.5-flash-latest": "Gemini Flash", 
        "nova-lite-v1": "Nova Lite",
        "llama-3.2-11b-vision-instruct": "Llama 3.2",
        "gemini-1.5-flash-latest-TEXT-NEW-PROMPT": "Gemini Flash (New)"
    }
    
    # Calculate max ground truth for setting axis limits - set to at least 18 if data allows
    max_value = max(df['ground_truth'].max() + 1, 18)
    
    # Draw a reference perfect prediction line
    plt.plot([0, max_value], [0, max_value], 'k--', alpha=0.3, label='Perfect prediction')
    
    # Draw mean lines for each model
    for model in models:
        # Get data for this model
        valid_data = df[['ground_truth', model]].dropna()
        
        if len(valid_data) == 0:
            continue
            
        # Calculate and plot trend line
        if len(valid_data) > 1:
            # Group by ground truth and calculate mean model answer
            grouped = valid_data.groupby('ground_truth')[model].mean().reset_index()
            
            # Sort by ground truth to ensure line is drawn correctly
            grouped = grouped.sort_values('ground_truth')
            
            # Plot the line
            plt.plot(grouped['ground_truth'], grouped[model], '-', color=colors[model], linewidth=3, label=short_names[model])
    
    # Set plot labels and title
    plt.xlabel('Ground Truth', fontsize=14)
    plt.ylabel('Model Answer', fontsize=14)
    plt.title('Model Performance: Ground Truth vs. Model Answers', fontsize=16)
    
    # Set axis limits
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Ensure no overlap
    plt.tight_layout()
    
    # Calculate and display RMSE for each model
    rmse_values = {}
    for model in models:
        valid_data = df[['ground_truth', model]].dropna()
        if len(valid_data) > 0:
            rmse = np.sqrt(((valid_data['ground_truth'] - valid_data[model]) ** 2).mean())
            rmse_values[model] = rmse
    
    # Add RMSE values as text box
    rmse_text = "RMSE Values:\n" + "\n".join([f"{short_names[model]}: {rmse:.2f}" for model, rmse in sorted(rmse_values.items(), key=lambda x: x[1])])
    plt.figtext(0.02, 0.02, rmse_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Display the plot
    plt.show()
    
    return rmse_values

def plot_max_in_frame_ratio(df, models):
    """
    Create simple mean line plots showing the ratio of max objects in a frame to model answers
    """
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Define colors for each model
    colors = {
        "gpt-4o": "#4285F4",
        "gemini-1.5-pro-latest": "#EA4335",
        "gemini-1.5-flash-latest": "#FBBC05", 
        "nova-lite-v1": "#34A853",
        "llama-3.2-11b-vision-instruct": "#8B4513",
        "gemini-1.5-flash-latest-TEXT-NEW-PROMPT": "#FF6D01"
    }
    
    # Define short names for models
    short_names = {
        "gpt-4o": "GPT-4o",
        "gemini-1.5-pro-latest": "Gemini Pro",
        "gemini-1.5-flash-latest": "Gemini Flash", 
        "nova-lite-v1": "Nova Lite",
        "llama-3.2-11b-vision-instruct": "Llama 3.2",
        "gemini-1.5-flash-latest-TEXT-NEW-PROMPT": "Gemini Flash (New)"
    }
    
    # Calculate the ratio for each model and add to dataframe
    for model in models:
        ratio_col = f"{model}_ratio"
        # Avoid division by zero
        df[ratio_col] = np.where(
            df[model] > 0, 
            df['max_number_in_single_frame'] / df[model], 
            np.nan
        )
    
    # Calculate max ground truth for setting axis limits - set to at least 18 if data allows
    max_value = max(df['ground_truth'].max() + 1, 18)
    
    # Draw a reference line at ratio = 1.0
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Ideal ratio (1.0)')
    
    # Draw mean lines for each model
    for model in models:
        ratio_col = f"{model}_ratio"
        
        # Get data for this model
        valid_data = df[['ground_truth', ratio_col]].dropna()
        
        if len(valid_data) == 0:
            continue
            
        # Calculate and plot trend line
        if len(valid_data) > 1:
            # Group by ground truth and calculate mean ratio
            grouped = valid_data.groupby('ground_truth')[ratio_col].mean().reset_index()
            
            # Sort by ground truth to ensure line is drawn correctly
            grouped = grouped.sort_values('ground_truth')
            
            # Plot the line
            plt.plot(grouped['ground_truth'], grouped[ratio_col], '-', color=colors[model], linewidth=3, label=short_names[model])
    
    # Set plot labels and title
    plt.xlabel('Ground Truth', fontsize=14)
    plt.ylabel('Max in Frame / Model Answer Ratio', fontsize=14)
    plt.title('Ratio of Maximum Objects in Frame to Model Answer', fontsize=16)
    
    # Set axis limits
    plt.xlim(0, max_value)
    plt.ylim(0, 2.5)  # Fixed limit at 2.5 as requested
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Ensure no overlap
    plt.tight_layout()
    
    # Calculate and display average ratios for each model
    avg_ratios = {}
    for model in models:
        ratio_col = f"{model}_ratio"
        valid_data = df[ratio_col].dropna()
        if len(valid_data) > 0:
            avg_ratio = valid_data.mean()
            avg_ratios[model] = avg_ratio
    
    # Add average ratio values as text box
    ratio_text = "Average Ratios:\n" + "\n".join([f"{short_names[model]}: {ratio:.2f}" for model, ratio in sorted(avg_ratios.items(), key=lambda x: x[1], reverse=True)])
    plt.figtext(0.02, 0.02, ratio_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add explanation text
    explanation = (
        "Ratio interpretation:\n" +
        "= 1.0: Model counts exactly what's visible in a frame\n" +
        "< 1.0: Model counts more than visible in any single frame\n" +
        "> 1.0: Model counts fewer than visible in a frame"
    )
    plt.figtext(0.65, 0.02, explanation, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Display the plot
    plt.show()
    
    return avg_ratios

# Run the analysis
if __name__ == "__main__":
    # Update this path to your full CSV file
    csv_path = "src/all_questions_aggregated_final.csv"
    analyze_number_questions(csv_path)