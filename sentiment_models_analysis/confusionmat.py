def plot_confusion_matrix(cm, model_name, ax):
 """Plot a 2×2 confusion matrix on the given Axes."""
 im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
 ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

 classes = ['Negative', 'Positive']
 tick_marks = np.arange(len(classes))
 ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=9)
 ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=9)

 thresh = cm.max() / 2.0
 for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
 ax.text(j, i, f'{cm[i, j]:,}',
 ha='center', va='center', fontsize=11,
 color='white' if cm[i, j] > thresh else 'black')

 ax.set_ylabel('True Label', fontsize=9)
 ax.set_xlabel('Predicted Label', fontsize=9)
 ax.set_title(model_name, fontsize=11, fontweight='bold')

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
 fig.patch.set_facecolor('#F9F9F9')

for ax, (model_name, cm) in zip(axes.flatten(), all_confusion.items()):
 ax.set_facecolor('#F9F9F9')
 plot_confusion_matrix(cm, model_name, ax)

fig.suptitle('Confusion Matrices — All Models', fontsize=15,
 fontweight='bold', y=1.01)
 plt.tight_layout()
 plt.savefig('/tmp/confusion_matrices.png', dpi=150, bbox_inches='tight')
 plt.show()
 print('Figure saved.')