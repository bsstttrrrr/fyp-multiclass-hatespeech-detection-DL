# Ran on Google Colab

import pandas as pd
import numpy as np
import re
import html
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Text Preprocessing & NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.chunk import ne_chunk

# Machine Learning & Deep Learning Frameworks
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, 
    LSTM, 
    Dense, 
    Dropout, 
    Conv1D, 
    GlobalMaxPool1D, 
    Bidirectional
)
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from transformers import TFDistilBertModel, DistilBertTokenizer

# Scikit-learn & Imblearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score
)
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --------------------------
# Utility: Safe AdamW import
# --------------------------
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
extra_stopwords = {"rt", "#ff", "ff"}
stop_words.update(extra_stopwords)



def preprocess(tweets):
    if not isinstance(tweets, pd.Series):
        tweets = pd.Series(tweets)

    # Your current steps
    tweets = tweets.str.replace(r'\bRT\b', '', regex=True)
    tweets = tweets.str.replace(r'@[\w\-]+', '', regex=True)
    tweets = tweets.str.replace(r'http\S+|www\.\S+', '', regex=True)
    tweets = tweets.str.replace(r'&\w+;', '', regex=True)
    tweets = tweets.str.replace(r'([!?])\1+', r'\1', regex=True)
    tweets = tweets.str.replace(r'^[\'"!:]+', '', regex=True)
    tweets = tweets.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Add lemmatization and stopwords
    def lemmatize_text(text):
        doc = nlp(text.lower())
        return ' '.join([token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha])

    tweets = tweets.apply(lemmatize_text)
    return tweets

def plot_training_history(history, model_name="Model"):
    """
    Plots training and validation loss/accuracy from a Keras History object.

    Args:
        history (History): The History object returned by model.fit().
        model_name (str): Name of the model for the plot title.
    """
    plt.figure(figsize=(8, 5))

    # Plot losses
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')

    # Plot accuracy if available
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')

    plt.legend()
    plt.title(f"{model_name} Training vs. Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
# ================ Data Preparation =================
def get_df():
    url = "https://raw.githubusercontent.com/NakulLakhotia/Hate-Speech-Detection-in-Social-Media-using-Python/refs/heads/master/HateSpeechData.csv"
    df = pd.read_csv(url)
    return df

def get_processed_df(df):
    tweet = df['tweet']
    processed_tweets = preprocess(tweet)
    df['processed_tweets'] = processed_tweets
    print(df[['tweet', 'processed_tweets']].head(10))
    return df
# =============== Train Test Split =================
def get_train_test_data(df):
    X = df['processed_tweets'].values
    y = df['class'].values
    # 60 train :40 temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    # 20 (val) : 20 (test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
# ============== Resampling =================
def get_resampled(X_train, y_train):
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train.reshape(-1, 1), y_train)
    
    visualize_resampled_df = pd.DataFrame({
        "text": X_train,   # first column
        "class": y_train
    })

    plt.figure(figsize=(8, 6))
    sns.countplot(x="class", data=visualize_resampled_df)
    plt.title("Distribution of Classes After Resampling")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()
    print(visualize_resampled_df["class"].value_counts())
    return X_train_resampled.flatten(), y_train_resampled

def get_class_weights(y_train):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class Weights:", class_weights_dict)
    return class_weights_dict

def get_Ablation_study_data(df):
    X_ablation = df['tweet'].values  # takes normal data
    y_ablation = df['class'].values
    # Step 1: Train + Temp split (80/20)
    X_train_ablation, X_temp_ablation, y_train_ablation, y_temp_ablation = train_test_split(
        X_ablation, y_ablation, test_size=0.4, random_state=42, stratify=y_ablation
    )
    # Step 2: Split temp into Validation + Test (50/50 of 40% = 20% each)
    X_val_ablation, X_test_ablation, y_val_ablation, y_test_ablation = train_test_split(
        X_temp_ablation, y_temp_ablation, test_size=0.5, random_state=42, stratify=y_temp_ablation
    )
    return X_train_ablation, X_val_ablation, X_test_ablation, y_train_ablation, y_val_ablation, y_test_ablation
    
# =============== Model Definition =================
def get_model_tokenizer():
    class TFDistilBertLayer(Layer):
        def __init__(self, model_name="distilbert-base-uncased", **kwargs):
            super(TFDistilBertLayer, self).__init__(**kwargs)
            self.bert = TFDistilBertModel.from_pretrained(model_name, from_pt=True)

        def call(self, inputs):
            input_ids, attention_mask = inputs
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

        def get_config(self):
            config = super().get_config()
            config.update({"model_name": "distilbert-base-uncased"})
            return config

    # ===== Load Tokenizer =====
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # ===== Custom Model with Hybrid Layers =====
    input_ids = tf.keras.Input(shape=(100,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(100,), dtype=tf.int32, name="attention_mask")

    # Use custom layer to get BERT outputs
    sequence_output = TFDistilBertLayer()(inputs=[input_ids, attention_mask])  # (batch, seq_len, hidden_size=768)

    # ---- CNN Branch ----
    cnn_out = Conv1D(filters=64, kernel_size=5, activation="relu")(sequence_output)
    cnn_out = GlobalMaxPool1D()(cnn_out)

    # ---- BiLSTM Branch ----
    lstm_out = Bidirectional(LSTM(64, return_sequences=False))(sequence_output)

    # ---- Merge CNN + LSTM ----
    merged = tf.keras.layers.concatenate([cnn_out, lstm_out])

    # ---- Dense Classifier ----
    dense = Dense(128, activation="relu")(merged)
    dropout = Dropout(0.5)(dense)
    output = Dense(3, activation="softmax")(dropout)

    # ===== Final Model =====
    distilBERT_CNN_BiLSTM = Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile
    distilBERT_CNN_BiLSTM.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    distilBERT_CNN_BiLSTM.summary()
    return distilBERT_CNN_BiLSTM, tokenizer

def tokenize_data(texts, tokenizer, max_length=100):
    encoded = tokenizer(
        texts.tolist(),  # Ensure list format
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encoded["input_ids"], encoded["attention_mask"]

def train_model (model , X_train, y_train, X_val, y_val, tokenizer,  epochs=30, batch_size=32, class_weights=None,):
    # Tokenize training and validation data
    train_input_ids, train_attention_mask = tokenize_data(X_train, tokenizer)
    val_input_ids, val_attention_mask = tokenize_data(X_val, tokenizer)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [train_input_ids, train_attention_mask],
        y_train,
        validation_data=([val_input_ids, val_attention_mask], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    return history

def evaluate_model(model, X_test, y_test, tokenizer, max_length=100, model_name="Model", plot_cm=True):
    """
    Evaluate a classification model and print classification report, kappa score, and confusion matrix.

    Parameters:
    -----------
    model : tf.keras.Model
        The trained model to evaluate.
    X_test : np.ndarray or list
        Test data (text strings).
    y_test : np.ndarray or list
        True labels (integers, e.g., 0, 1, 2 for 3-class classification).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for preprocessing text (e.g., DistilBertTokenizer).
    max_length : int, optional (default=100)
        Maximum sequence length for tokenization.
    model_name : str, optional (default="Model")
        Name of the model for printing purposes.
    plot_cm : bool, optional (default=True)
        Whether to plot the confusion matrix using seaborn.

    Returns:
    --------
    dict : Evaluation metrics (loss, accuracy, classification report, kappa score, confusion matrix).
    """
    # Clean text (decode HTML entities)
    X_test_cleaned = [html.unescape(text) for text in X_test]
    # Tokenize with explicit max_length
    test_input_ids, test_attention_mask = tokenize_data(X_test_cleaned, max_length)

    # Verify input shapes
    print(f"Input IDs shape: {test_input_ids.shape}")
    print(f"Attention Mask shape: {test_attention_mask.shape}")

    y_test = np.array(y_test, dtype=np.int32)

    # Evaluate model
    loss, accuracy = model.evaluate([test_input_ids, test_attention_mask], y_test, verbose=0)
    print(f"\n{model_name} Evaluation:")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict
    predictions = model.predict([test_input_ids, test_attention_mask], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Classification report
    class_report = classification_report(y_test, predicted_classes, digits=4)
    print("\nClassification Report:")
    print(class_report)

    # Cohen's kappa score
    kappa = cohen_kappa_score(y_test, predicted_classes)
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix (if enabled)
    if plot_cm:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # Return metrics for further analysis
    return {
        "loss": loss,
        "accuracy": accuracy,
        "classification_report": class_report,
        "kappa_score": kappa,
        "confusion_matrix": cm
    }