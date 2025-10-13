# Developing a Deep Learning Model for Social Media Hate Speech Text Detection

### Final Year Project (Bachelor of Computer Science)

This project investigates **multiclass hate speech detection** with a focus on improving the performance of the **minority class (Hate Speech – label 0)**.  
The study integrates **DistilBERT embeddings** with a hybrid **CNN–BiLSTM** model and includes a detailed **ablation study** to evaluate the effects of different preprocessing techniques on classification performance.

---

## Problem Statement

Despite significant progress in NLP-based text classification, detecting **hate speech** remains a challenge — particularly in **imbalanced datasets** where hate speech examples are rare.  
This project addresses the following key issues:

- Underperformance on the **minority class (hate speech)** in multiclass classification.  
- The impact of **data preprocessing techniques** on model robustness and accuracy.  
- The effectiveness of **hybrid deep learning architectures** combining CNN and BiLSTM on top of DistilBERT embeddings.

#### Data Imbalance

![imba](https://github.com/bsstttrrrr/fyp-multiclass-hatespeech-detection-DL/blob/main/dist.png?raw=true)
---

##  Project Pipeline


1. **Data Loading & Preprocessing**  
   - Text cleaning, lemmatization, stopword removal, tokenization, and normalization.  
2. **Feature Extraction**  
   - Generating contextual embeddings using **DistilBERT**.  
3. **Model Construction**  
   - Hybrid **CNN–BiLSTM** network for sequence classification.  
4. **Evaluation & Ablation Study**  
   - Comparing results across preprocessing variants and architectural configurations.

![pipeline](https://github.com/bsstttrrrr/fyp-multiclass-hatespeech-detection-DL/blob/main/fyp2flow.png?raw=true)

---

## Model Architecture

The proposed model architecture combines:

- **DistilBERT Encoder** → Generates dense contextual embeddings.  
- **CNN Layer** → Extracts local semantic features.  
- **BiLSTM Layer** → Captures long-range contextual dependencies.  
- **Fully Connected Layers** → Perform final classification into three classes:  
  *Hate Speech (0), Offensive (1), Neutral (2)*.

![archi](https://github.com/bsstttrrrr/fyp-multiclass-hatespeech-detection-DL/blob/main/archi.png?raw=true)

---

## 🧪 Implementation and Experimental Results

The evaluation includes:

- **Ablation study** on preprocessing techniques and model configurations.
- **Resampling** through oversampling minority classes
- **Weighted Training** to penalize misclassifications
- **Metrics:** Accuracy, Precision, Recall, and F1-score (macro/weighted).  
- Focused analysis on **minority-class recall and F1-score**.  
- Comparison against **state-of-the-art (SOTA)** results.

#### Resampled Distribution
![resampled](https://github.com/bsstttrrrr/fyp-multiclass-hatespeech-detection-DL/blob/main/resam.png?raw=true)

#### Experimental Results
![f1result](https://github.com/bsstttrrrr/fyp-multiclass-hatespeech-detection-DL/blob/main/res1.png?raw=true)
![compresult](https://github.com/bsstttrrrr/fyp-multiclass-hatespeech-detection-DL/blob/main/res2.png?raw=true)

---

## Implementation Details

- **Dataset:** Davidson et al. (2017) Twitter Hate Speech Dataset  
- **Frameworks:** Tensorflow, Hugging Face Transformers  
- **Language:** Python 3.10  
- **Hardware:** CUDA-enabled GPU  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  

---

## Key Contributions

- Developed a **DistilBERT + CNN–BiLSTM hybrid model** for multiclass hate speech detection.  
- Conducted a **comprehensive ablation study** on preprocessing strategies.  
- Improved **minority-class performance** while maintaining overall accuracy.  
- Benchmarked results against **state-of-the-art approaches** for comparison.

---

### Acknowledgement
This project is conducted under the supervision of Dr Ramesh Kumar Ayyasamy (UTAR),
and credit of ablation studies due to Dr Bashar Tahayna (UTAR).

### Reference
Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM. 

 
