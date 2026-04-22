VOCAB_SIZE   = 20_000   
MAX_LEN      = 256      
EMBED_DIM    = 64      
LSTM_UNITS   = 64
BATCH_SIZE   = 128
EPOCHS       = 10
MAX_TFIDF    = 20_000   

# VOCAB_SIZE (20,000)
# → Only the top 20,000 most frequent words are kept
# → Reduces noise and computation
# MAX_LEN (256)
# → Each review is limited to 256 words (tokens)
# → Longer reviews are cut, shorter ones are padded
# EMBED_DIM (64)
# → Each word is converted into a 64-dimensional vector
# → Used in the embedding layer (LSTM model)
# LSTM_UNITS (64)
# → Number of neurons in the LSTM layer
# → Controls model complexity
# BATCH_SIZE (128)
# → Number of samples processed at once during training
# EPOCHS (10)
# → Number of times the model sees the full dataset
# MAX_TFIDF (20,000)
# → Maximum number of features for TF-IDF (used in ML models)

print('Loading IMDB dataset (integer sequences) …')
(X_train_seq, y_train), (X_test_seq, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print(f'  Train samples : {len(X_train_seq):,}')
print(f'  Test  samples : {len(X_test_seq):,}')
print(f'  Label balance (train) – pos: {y_train.sum():,}  neg: {(1-y_train).sum():,}')

# Counts:
# Positive reviews (1) → y_train.sum()
# Negative reviews (0) → (1 - y_train).sum()
# checks if the dataset is balanced
