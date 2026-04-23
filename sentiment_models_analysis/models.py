# ════════════════════════════════════════════════════════════════════════════
# Build LSTM model
# Architecture:
#   Embedding → LSTM → Dropout → Dense(sigmoid)
# ════════════════════════════════════════════════════════════════════════════

def build_lstm(vocab_size, embed_dim, lstm_units, max_len):
    model = Sequential([

        # Embedding Layer:
        # Converts word indices into dense vectors of fixed size (embed_dim)
        # Input shape: (batch_size, max_len)
        # Output shape: (batch_size, max_len, embed_dim)
        Embedding(input_dim=vocab_size, output_dim=embed_dim,
                  input_length=max_len, name='embedding'),

        # LSTM Layer:
        # Processes the sequence of embeddings and captures temporal dependencies
        # return_sequences=False → only final hidden state is used (for classification)
        LSTM(lstm_units, return_sequences=False, name='lstm'),

        # Dropout Layer:
        # Randomly drops 30% of neurons during training to reduce overfitting
        Dropout(0.3, name='dropout'),

        # Output Layer:
        # Single neuron with sigmoid activation for binary classification
        Dense(1, activation='sigmoid', name='output')
    ], name='LSTM_Sentiment')

    # Compile the model:
    # - optimizer='adam' → adaptive learning rate optimization
    # - loss='binary_crossentropy' → suitable for binary classification
    # - metrics=['accuracy'] → track accuracy during training
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Build the model using predefined hyperparameters
lstm_model = build_lstm(VOCAB_SIZE, EMBED_DIM, LSTM_UNITS, MAX_LEN)

# Print model architecture summary
lstm_model.summary()


# ── Train LSTM ─────────────────────────────────────────────────────────────

# EarlyStopping:
# Stops training if validation loss does not improve for 2 epochs
# restore_best_weights=True → keeps the best model instead of last
early_stop = EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)

print('Training LSTM …')
t0 = time.time()

# Train the model
lstm_history = lstm_model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,   # 10% of training data used for validation
    callbacks=[early_stop],
    verbose=1
)

print(f'Training time: {time.time()-t0:.1f}s')


# ── Evaluate LSTM ──────────────────────────────────────────────────────────

# Predict probabilities (values between 0 and 1)
lstm_probs = lstm_model.predict(X_test_pad, verbose=0).ravel()

# Convert probabilities to binary predictions using threshold = 0.5
lstm_preds = (lstm_probs >= 0.5).astype(int)

# Compute evaluation metrics (accuracy, precision, recall, etc.)
res_lstm = compute_metrics(y_test, lstm_preds, 'LSTM')

# Store results
all_results.append(res_lstm)
all_confusion['LSTM'] = confusion_matrix(y_test, lstm_preds)

print('\nClassification Report:')
print(classification_report(y_test, lstm_preds,
      target_names=['Negative', 'Positive']))


# ════════════════════════════════════════════════════════════════════════════
# LinearSVC — efficient linear SVM for large sparse TF-IDF matrices
# ════════════════════════════════════════════════════════════════════════════

print('Training SVM (LinearSVC) …')
t0 = time.time()

# Initialize Linear Support Vector Classifier
# C → regularization strength (higher = less regularization)
# max_iter → max training iterations
svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=SEED)

# Train using TF-IDF features
svm_model.fit(X_train_tfidf, y_train)

print(f'Training time: {time.time()-t0:.1f}s')

# Predict class labels
svm_preds = svm_model.predict(X_test_tfidf)

# Evaluate performance
res_svm = compute_metrics(y_test, svm_preds, 'SVM')
all_results.append(res_svm)
all_confusion['SVM'] = confusion_matrix(y_test, svm_preds)

print('\nClassification Report:')
print(classification_report(y_test, svm_preds,
      target_names=['Negative', 'Positive']))


# ════════════════════════════════════════════════════════════════════════════
# MultinomialNB — requires non-negative features (TF-IDF values are ≥ 0)
# ════════════════════════════════════════════════════════════════════════════

print('Training Naive Bayes (MultinomialNB) …')
t0 = time.time()

# Initialize Naive Bayes classifier
# alpha → smoothing parameter (prevents zero probabilities)
nb_model = MultinomialNB(alpha=0.1)

# Train model
nb_model.fit(X_train_tfidf, y_train)

print(f'Training time: {time.time()-t0:.1f}s')

# Predict labels
nb_preds = nb_model.predict(X_test_tfidf)

# Evaluate performance
res_nb = compute_metrics(y_test, nb_preds, 'Naive Bayes')
all_results.append(res_nb)
all_confusion['Naive Bayes'] = confusion_matrix(y_test, nb_preds)

print('\nClassification Report:')
print(classification_report(y_test, nb_preds,
      target_names=['Negative', 'Positive']))


# ════════════════════════════════════════════════════════════════════════════
# Logistic Regression — lbfgs solver; L2 regularisation
# ════════════════════════════════════════════════════════════════════════════

print('Training Logistic Regression …')
t0 = time.time()

# Initialize Logistic Regression model
# C → inverse of regularization strength
# solver='lbfgs' → good default for small/medium datasets
lr_model = LogisticRegression(C=5.0, max_iter=1000,
                              solver='lbfgs', random_state=SEED)

# Train model
lr_model.fit(X_train_tfidf, y_train)

print(f'Training time: {time.time()-t0:.1f}s')

# Predict labels
lr_preds = lr_model.predict(X_test_tfidf)

# Evaluate performance
res_lr = compute_metrics(y_test, lr_preds, 'Logistic Regression')
all_results.append(res_lr)
all_confusion['Logistic Regression'] = confusion_matrix(y_test, lr_preds)

print('\nClassification Report:')
print(classification_report(y_test, lr_preds,
      target_names=['Negative', 'Positive']))
