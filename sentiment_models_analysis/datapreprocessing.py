print('Decoding integer sequences to text …')

word_index   = imdb.get_word_index()

idx_to_word  = {v + 3: k for k, v in word_index.items()}
idx_to_word.update({0: '<PAD>', 1: '<START>', 2: '<UNK>', 3: '<UNUSED>'})
# <PAD> → padding (extra zeros added)
# <START> → beginning of review
# <UNK> → unknown word
# <UNUSED> → not used
def decode_review(seq):
    """Convert integer sequence back to human-readable string."""
    return ' '.join(idx_to_word.get(i, '?') for i in seq)

X_train_text = [decode_review(seq) for seq in X_train_seq]
X_test_text  = [decode_review(seq) for seq in X_test_seq]

print('Sample decoded review (first 120 chars):') 
print('  ', X_train_text[0][:120], '…')

# Prints:

# First review
# Only first 120 characters (to keep it short)
# Originally:

# Data = numbers (for model training)

# Now:

# Data = readable sentences (for humans)

# This step is mainly for:

# Understanding the dataset
# Debugging
# Visualization




def clean_text(text: str) -> str:
    """
    Clean a raw review string:
    1. Lowercase
    2. Remove IMDB special tokens
    3. Remove punctuation / digits
    4. Strip extra whitespace
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          
    text = re.sub(r'[^a-z\s]', ' ', text)         
    text = re.sub(r'\s+', ' ', text).strip()       
    return text

# Replaces multiple spaces with a single space
# Removes spaces from start/end

print('Cleaning text …')
X_train_clean = [clean_text(t) for t in X_train_text]
X_test_clean  = [clean_text(t) for t in X_test_text]

print('Sample cleaned review (first 120 chars):')
print('  ', X_train_clean[0][:120], '…')

#   Text cleaning
#      – lowercase
#      – remove punctuation & digits
#      – collapse extra whitespace
#      – remove IMDB placeholder tokens (<PAD>, <START>, etc.)


# Removes noise (symbols, junk)
# Makes text consistent
# Improves performance of:
# TF-IDF
# Machine Learning models
# Deep Learning models





print('Fitting TF-IDF vectorizer …')
tfidf = TfidfVectorizer(
    max_features = MAX_TFIDF,
    ngram_range  = (1, 2),          
    sublinear_tf = True,           
    min_df       = 3,               
    strip_accents = 'unicode',
    analyzer     = 'word'
)
# TF (Term Frequency) → how often word appears in a review
# IDF (Inverse Document Frequency) → how rare the word is across all reviews
# X_train_tfidf = tfidf.fit_transform(X_train_clean)
# X_test_tfidf  = tfidf.transform(X_test_clean)

print(f'  TF-IDF matrix shape  – train: {X_train_tfidf.shape}')
print(f'  TF-IDF matrix shape  – test : {X_test_tfidf.shape}')

# TF-IDF feature extraction  (SVM · Naive Bayes · Logistic Regression)

# max_features = 20000
# → Keep only top 20,000 important words/features
# ngram_range = (1, 2)
# → Use:
# single words → "good"
# pairs of words → "very good"
#  Helps capture context
# sublinear_tf = True
# → Uses 1 + log(tf) instead of raw counts
#  Reduces impact of very frequent words
# min_df = 3
# → Ignore words that appear in fewer than 3 documents
#  Removes rare/noisy words
# strip_accents = 'unicode'
# → Converts accented characters to normal form
# analyzer = 'word'
# → Works at word level (not characters)

# Text → TF-IDF → Numbers → Model input






print('Padding sequences for LSTM …')
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

print(f'  Padded train shape : {X_train_pad.shape}')
print(f'  Padded test  shape : {X_test_pad.shape}')

# LSTM reads sequences step-by-step
# Needs consistent input size
# Padding ensures:
# Proper batch processing
# Efficient training

# Variable-length text → padded to fixed length → fed into LSTM