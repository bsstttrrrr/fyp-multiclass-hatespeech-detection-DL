# Ran on Google Colab

# ================== Imports ==================
import pandas as pd
import numpy as np
import re, html
import matplotlib.pyplot as plt
import seaborn as sns
# NLP
# import nltk
# from nltk.corpus import stopwords
# import spacy
# nltk.download('stopwords')
# nlp = spacy.load("en_core_web_sm")
# stop_words = set(stopwords.words("english"))
# stop_words.update({"rt", "#ff", "ff"})
from tensorflow.keras import layers
# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Layer, Dense, Dropout, LSTM, Conv1D, GlobalMaxPool1D, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Transformers
from transformers import TFDistilBertModel, DistilBertTokenizer

# Metrics & utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import keras_tuner as kt
from imblearn.over_sampling import RandomOverSampler
# --------------------------
# Utility: Safe AdamW import
# --------------------------
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.models import load_model
# nltk.download('stopwords')
# nlp = spacy.load("en_core_web_sm")
# stop_words = set(stopwords.words("english"))
# extra_stopwords = {"rt", "#ff", "ff"}
# stop_words.update(extra_stopwords)

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import gc, torch
from tensorflow.keras import backend as K

# def preprocess(tweets):
#     if not isinstance(tweets, pd.Series):
#         tweets = pd.Series(tweets)

#     # Your current steps
#     tweets = tweets.str.replace(r'\bRT\b', '', regex=True)
#     tweets = tweets.str.replace(r'@[\w\-]+', '', regex=True)
#     tweets = tweets.str.replace(r'http\S+|www\.\S+', '', regex=True)
#     tweets = tweets.str.replace(r'&\w+;', '', regex=True)
#     tweets = tweets.str.replace(r'([!?])\1+', r'\1', regex=True)
#     tweets = tweets.str.replace(r'^[\'"!:]+', '', regex=True)
#     tweets = tweets.str.replace(r'\s+', ' ', regex=True).str.strip()

#     # Add lemmatization and stopwords
#     def lemmatize_text(text):
#         doc = nlp(text.lower())
#         return ' '.join([token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha])

#     tweets = tweets.apply(lemmatize_text)
#     return tweets

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
# def get_df():
#     url = "https://raw.githubusercontent.com/NakulLakhotia/Hate-Speech-Detection-in-Social-Media-using-Python/refs/heads/master/HateSpeechData.csv"
#     df = pd.read_csv(url)
#     return df

# def get_processed_df(df):
#     tweet = df['tweet']
#     processed_tweets = preprocess(tweet)
#     df['processed_tweets'] = processed_tweets
#     print(df[['tweet', 'processed_tweets']].head(10))
#     return df
# =============== Train Test Split =================
def get_train_test_data(df):
    X = df['processed_tweets'].astype(str).tolist()
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

    # Wrap X_train in a DataFrame so it's 2D (n,1)
    X_train_resampled, y_train_resampled = ros.fit_resample(
        pd.DataFrame(X_train), y_train
    )

    # Flatten back to 1D text array
    X_train_resampled = X_train_resampled[0].values  

    # Visualization
    visualize_resampled_df = pd.DataFrame({
        "text": X_train_resampled,
        "class": y_train_resampled
    })

    plt.figure(figsize=(8, 6))
    sns.countplot(x="class", data=visualize_resampled_df)
    plt.title("Distribution of Classes After Resampling")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    print(visualize_resampled_df["class"].value_counts())
    return X_train_resampled, y_train_resampled


def get_class_weights(y_train):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class Weights:", class_weights_dict)
    return class_weights_dict

def get_ablation_study_data(df):
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
    input_ids = tf.keras.Input(shape=(52,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(52,), dtype=tf.int32, name="attention_mask")

    # Use custom layer to get BERT outputs
    sequence_output = TFDistilBertLayer()(inputs=[input_ids, attention_mask])  # (batch, seq_len, hidden_size=768)

    # ---- CNN Branch ----
    # ===== CNN Branch with Multiple Kernels =====
    conv3 = GlobalMaxPool1D()(Conv1D(64, kernel_size=3, activation="relu", padding="same")(sequence_output))
    conv5 = GlobalMaxPool1D()(Conv1D(64, kernel_size=5, activation="relu", padding="same")(sequence_output))
    conv7 = GlobalMaxPool1D()(Conv1D(64, kernel_size=7, activation="relu", padding="same")(sequence_output))
    cnn_out = tf.keras.layers.concatenate([conv3, conv5, conv7])  # Combine features
    cnn_out = tf.keras.layers.Concatenate()([conv3, conv5, conv7])
    # ===== BiLSTM Branch =====
    lstm_out = Bidirectional(LSTM(64, return_sequences=False))(sequence_output)
    merged = tf.keras.layers.Concatenate()([cnn_out, lstm_out])

    # ---- Dense Classifier ----
    dense_out = Dense(256, activation="relu")(merged)
    dense_out = Dropout(0.3)(dense_out)
    dense_out = Dense(128, activation="relu")(dense_out)
    dense_out = Dropout(0.3)(dense_out)
    output = Dense(3, activation="softmax")(dense_out)

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

def tokenize_data(texts, tokenizer, max_length=52):
    if not isinstance(texts, list):
        texts = texts.tolist()
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encoded["input_ids"], encoded["attention_mask"]

def train_model (model , X_train, y_train, X_val, y_val, tokenizer,  epochs, batch_size, class_weights=None,):
    # Tokenize training and validation data
    train_input_ids, train_attention_mask = tokenize_data(X_train, tokenizer)
    val_input_ids, val_attention_mask = tokenize_data(X_val, tokenizer)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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

def evaluate_model(model, X_test, y_test, tokenizer, max_length=52, model_name="Model", plot_cm=True):
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
    test_input_ids, test_attention_mask = tokenize_data(X_test_cleaned, tokenizer, max_length)

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

def plot_multiclass_roc(model, X_val, y_val, tokenizer, classes=[0, 1, 2], max_length=52, model_name="Model"):
    """
    Compute and plot multiclass ROC curves with macro-average AUC.
    Accepts raw text X_val and tokenizes internally.

    Parameters:
    -----------
    model : tf.keras.Model
        Trained Keras model.
    X_val : list or np.ndarray
        Raw validation text data.
    y_val : np.ndarray
        True labels (integer encoded).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for preprocessing text (e.g., DistilBertTokenizer).
    classes : list
        List of class labels.
    max_length : int
        Maximum sequence length for tokenization.
    model_name : str
        Name of the model for plot title.

    Returns:
    --------
    float : Macro-average AUC score
    """
    # Tokenize X_val using your tokenize_data function
    X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer, max_length=max_length)

    # Predict probabilities
    y_pred_probs = model.predict([X_val_ids, X_val_mask], verbose=0)

    # Binarize true labels
    y_val_bin = label_binarize(y_val, classes=classes)

    # Compute macro-average AUC
    auc_score = roc_auc_score(y_val_bin, y_pred_probs, average="macro", multi_class="ovr")
    print(f"{model_name} Testing AUC (macro): {auc_score:.4f}")

    # Plot ROC curves per class
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} Multiclass ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc_score

# Experimentation 
def clear_cache():
    """Free TF + Torch GPU memory."""
    K.clear_session()
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
def run_experiment(df, exp_name, get_data_fn, resample=False, use_class_weights=False, epochs=30, batch_size=16):
    print(f"\n{'='*20} {exp_name} {'='*20}")

    X_train, X_val, X_test, y_train, y_val, y_test = get_data_fn(df)
    if resample:
        X_train, y_train = get_resampled(X_train, y_train)

    # Apply class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = get_class_weights(y_train)

    # Get model + tokenizer
    model, tokenizer = get_model_tokenizer()

    # Train
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        class_weights=class_weights
    )

    plot_training_history(history, model_name=exp_name)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, tokenizer, model_name=exp_name)

    # ROC curve
    auc_score = plot_multiclass_roc(model, X_val, y_val, tokenizer, model_name=exp_name)

    clear_cache()
    return {"results": results, "auc_score": auc_score}

def collect_results(all_results):
    """
    Convert dictionary of experiment results into a DataFrame for visualization.
    all_results: dict -> {"ExperimentName": {"results": ..., "auc_score": ...}, ...}
    """
    records = []

    for exp_name, exp_data in all_results.items():
        results = exp_data["results"]  # from evaluate_model
        auc_score = exp_data["auc_score"]

        # Extract the key metrics you care about
        records.append({
            "Experiment": exp_name,
            "Accuracy": results.get("accuracy", None),
            "Kappa": results.get("kappa", None),
            "AUC": auc_score
        })

    return pd.DataFrame(records)