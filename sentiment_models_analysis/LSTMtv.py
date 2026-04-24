hist = lstm_history.history
 epochs_ran = range(1, len(hist['accuracy']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
 fig.patch.set_facecolor('#F9F9F9')

# ── Accuracy ─────────────────────────────────────────────────────────────
 ax1.set_facecolor('#F9F9F9')
 ax1.plot(epochs_ran, hist['accuracy'], color='#2E86AB', lw=2,
 marker='o', markersize=5, label='Train Accuracy')
 ax1.plot(epochs_ran, hist['val_accuracy'], color='#C73E1D', lw=2,
 marker='s', markersize=5, linestyle='--', label='Val Accuracy')
 ax1.set_title('LSTM — Accuracy per Epoch', fontsize=13, fontweight='bold')
 ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
 ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
 ax1.legend(); ax1.grid(linestyle='--', alpha=0.4)
 ax1.spines[['top','right']].set_visible(False)

# ── Loss ─────────────────────────────────────────────────────────────────
 ax2.set_facecolor('#F9F9F9')
 ax2.plot(epochs_ran, hist['loss'], color='#2E86AB', lw=2,
 marker='o', markersize=5, label='Train Loss')
 ax2.plot(epochs_ran, hist['val_loss'], color='#C73E1D', lw=2,
 marker='s', markersize=5, linestyle='--', label='Val Loss')
 ax2.set_title('LSTM — Loss per Epoch', fontsize=13, fontweight='bold')
 ax2.set_xlabel('Epoch'); ax2.set_ylabel('Binary Cross-Entropy Loss')
 ax2.legend(); ax2.grid(linestyle='--', alpha=0.4)
 ax2.spines[['top','right']].set_visible(False)

fig.suptitle('LSTM Training History', fontsize=15, fontweight='bold', y=1.01)
 plt.tight_layout()
 plt.savefig('/tmp/lstm_history.png', dpi=150, bbox_inches='tight')
 plt.show()
 print('Figure saved.')