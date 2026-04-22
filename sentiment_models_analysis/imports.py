import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED) 

# random generation, random results every time code compiles

import random
random.seed(SEED)

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Embedding: converts words to vectors
# LSTM: handles sequential/text data
# Dense: fully connected layer
# Dropout: prevents overfitting

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)

print('All imports successful.')
print(f'   TensorFlow  : {tf.__version__}')
print(f'   NumPy       : {np.__version__}')
print(f'   Pandas      : {pd.__version__}')

# re: text processing (regular expressions)
# time: measuring execution time
# warnings: suppress unnecessary warnings
# numpy: numerical computations
# pandas: data handling (tables/dataframes)
# matplotlib: data visualization (graphs)
