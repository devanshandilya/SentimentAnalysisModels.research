def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    """
    This function calculates common evaluation metrics for a binary classification model.

    Parameters:
    - y_true: Actual/ground truth labels
    - y_pred: Predicted labels from the model
    - model_name: Name of the model (just for display)

    Returns:
    - A dictionary containing all computed metrics
    """

    # Step 1: Compute evaluation metrics

    # Accuracy → Overall correctness of the model
    # Formula: (TP + TN) / Total predictions
    acc = accuracy_score(y_true, y_pred)

    # Precision → Out of all predicted positives, how many were correct
    # Formula: TP / (TP + FP)
    # zero_division=0 prevents errors when denominator becomes 0
    prec = precision_score(y_true, y_pred, zero_division=0)

    # Recall → Out of all actual positives, how many were captured
    # Formula: TP / (TP + FN)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # F1 Score → Balance between Precision and Recall
    # Formula: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Step 2: Print results in a clean format

    # Print a separator line for readability
    print(f'\n{"─"*50}')

    # Display model name
    print(f'  Model      : {model_name}')

    # Another separator
    print(f'{"─"*50}')

    # Print each metric along with its formula (helps in explanation)
    print(f'  Accuracy   = (TP+TN)/(TP+TN+FP+FN)  = {acc:.4f}')
    print(f'  Precision  = TP/(TP+FP)              = {prec:.4f}')
    print(f'  Recall     = TP/(TP+FN)              = {rec:.4f}')
    print(f'  F1-Score   = 2×P×R/(P+R)             = {f1:.4f}')

    # Step 3: Return results as a dictionary

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }


# Storage for multiple models' results

# List to store metrics from different models
all_results = []

# Dictionary to store confusion matrices for each model
# Format: { model_name: confusion_matrix }
all_confusion = {}

print('✅ Metrics helper defined.')
