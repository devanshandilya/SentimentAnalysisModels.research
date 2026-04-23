# Define colors for each metric (used in the bar chart)
COLORS  = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# List of evaluation metrics we want to plot
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Get model names from the DataFrame index
MODELS  = df_results.index.tolist()

# Create x-axis positions (0, 1, 2, ... for each model)
x = np.arange(len(MODELS))

# Width of each bar (since we’ll have multiple bars per model)
width = 0.18


# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set a light background color for better visuals
fig.patch.set_facecolor('#F9F9F9')
ax.set_facecolor('#F9F9F9')


# Loop through each metric and plot its bars
for i, (metric, color) in enumerate(zip(METRICS, COLORS)):
    
    # Get values of the current metric for all models
    vals = df_results[metric].values
    
    # Calculate offset so bars for different metrics appear side-by-side
    offset = (i - 1.5) * width
    
    # Plot bars for this metric
    bars = ax.bar(x + offset, vals, width,
                  label=metric,              # legend label
                  color=color,               # bar color
                  alpha=0.88,                # slight transparency
                  edgecolor='white',         # border color
                  linewidth=0.6)
    
    # Add value labels above each bar
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,   # center of bar
                bar.get_height() + 0.003,          # slightly above bar
                f'{v:.3f}',                        # value with 3 decimals
                ha='center', va='bottom',
                fontsize=7.5, color='#333')


# Set x-axis tick positions and labels (model names)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=11)

# Set y-axis range (zoomed into high-performance region)
ax.set_ylim(0.70, 1.01)

# Format y-axis as percentage (e.g., 0.95 → 95%)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# Add axis label and chart title
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — All Metrics',
             fontsize=14, fontweight='bold', pad=14)

# Add legend to identify each metric
ax.legend(loc='lower right', fontsize=10, framealpha=0.7)

# Add horizontal grid lines for easier comparison
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Remove top and right borders for a cleaner look
ax.spines[['top','right']].set_visible(False)


# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure as an image file
plt.savefig('/tmp/comparison_grouped.png',
            dpi=150, bbox_inches='tight')

# Display the plot
plt.show()

# Confirmation message
print('Figure saved.')

# Create a 2x2 grid of subplots (one for each metric)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Set overall figure background color
fig.patch.set_facecolor('#F9F9F9')

# Flatten the 2D axes array into 1D for easy looping
axes_flat = axes.flatten()

# Define color palette for bars (one color per model)
palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']


# Loop through each subplot and corresponding metric
for ax, metric in zip(axes_flat, METRICS):
    
    # Set background color for each subplot
    ax.set_facecolor('#F9F9F9')
    
    # Get metric values for all models
    vals = df_results[metric].values
    
    # Create bar chart for current metric
    bars = ax.bar(MODELS, vals,
                  color=palette,        # different color per model
                  alpha=0.88,
                  edgecolor='white',
                  linewidth=0.8)
    
    # Add value labels on top of each bar
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,   # center of bar
                bar.get_height() + 0.002,          # slightly above bar
                f'{v:.4f}',                        # value with 4 decimals
                ha='center', va='bottom',
                fontsize=9, color='#222')
    
    # Set title of each subplot (metric name)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    
    # Set y-axis limits (focus on high scores)
    ax.set_ylim(0.70, 1.02)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    
    # Adjust x-axis label size
    ax.tick_params(axis='x', labelsize=9)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Remove top and right borders for cleaner look
    ax.spines[['top','right']].set_visible(False)


# Add an overall title for the entire figure
fig.suptitle('IMDB Sentiment Analysis — Per-Metric Comparison',
             fontsize=15, fontweight='bold', y=1.01)

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Save the figure as an image file
plt.savefig('/tmp/comparison_subplots.png',
            dpi=150, bbox_inches='tight')

# Display the plots
plt.show()

# Confirmation message
print('Figure saved.')

