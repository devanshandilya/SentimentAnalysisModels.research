# Convert the list of results into a pandas DataFrame
# Each row represents a model and its evaluation metrics
df_results = pd.DataFrame(all_results)

# Set 'Model' column as index so models appear as row labels
df_results = df_results.set_index('Model')

# Select only the important evaluation metrics in a fixed order
df_results = df_results[['Accuracy', 'Precision', 'Recall', 'F1-Score']]

# Print a header to clearly separate the output section
print('\n' + '='*60)
print('        FINAL COMPARISON TABLE (all metrics × models)')
print('='*60)

# Create a copy of the DataFrame just for display purposes
# (so we don't change the original numeric values)
display_df = df_results.copy()

# Convert all values to 4 decimal places for neat printing
# This improves readability when presenting results
for col in display_df.columns:
    display_df[col] = display_df[col].map(lambda x: f'{x:.4f}')

# Print the formatted table
print(display_df.to_string())
print('='*60)

# Now we find the best model for each metric
print('Best model per metric:')

# Loop through each metric column (Accuracy, Precision, etc.)
for col in df_results.columns:
    
    # idxmax() gives the model name with the highest value
    best = df_results[col].idxmax()
    
    # df_results[col].max() gives the highest score itself
    print(f'  {col:12s}: {best}  ({df_results[col].max():.4f})')