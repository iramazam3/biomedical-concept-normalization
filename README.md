# A Hybrid LLM and Embedding-Based Approach for Biomedical Concept Normalization in Clinical and Social Media Data

## 📌 Overview

Biomedical Concept Normalization (BCN) is a key task in biomedical NLP that maps diverse health-related expressions to standardized concepts such as UMLS CUIs. This project presents a unified hybrid framework that integrates Large Language Models (LLMs) with ontology-aware biomedical embeddings to handle linguistic variability across clinical and social media data.

The system combines LLM-based linguistic normalization with SapBERT embeddings and FAISS-based retrieval to enable accurate and scalable normalization from both structured clinical text and noisy, unstructured social media posts.

---

## 🎯 Objectives

* Normalize health-related text to standardized UMLS concepts
* Handle linguistic variability across domains (clinical vs social media)
* Support both:

  * **BCN (Biomedical Concept Normalization)** – phrase-level mapping
  * **BEL (Biomedical Entity Linking)** – full-text processing

---

## 📂 Datasets

### 1. Clinical Dataset (BCN)

* 3,144 EHR-derived symptom phrases
* Pre-annotated with UMLS CUIs
* Task: phrase-level normalization

### 2. Social Media Phrase Dataset (BCN)

* 102 Twitter-derived symptom expressions
* Informal and noisy language

### 3. Twitter Full-Text Dataset (BEL)

* 600 tweets
* Includes:

  * Personal Health Experience (PHE) classification
  * Symptom extraction
  * Concept normalization
    
* Not uploaded due to Twitter's data usage and redistribution restrictions

### 4. Kaggle COVID-19 Dataset (Validation)

* Real-world tweets with hashtags (#covid, #quarantine, etc.)
* Used for end-to-end pipeline validation

* Not uploaded due to Twitter's data usage and redistribution restrictions

---

## 🚀 Features

* Unified framework for **clinical + social media data**
* LLM-based linguistic normalization (informal → medical terms)
* SapBERT embeddings for semantic similarity
* FAISS for efficient large-scale retrieval
* Supports both BCN and BEL tasks
* Robust handling of synonym variability and paraphrases

---

## 🧠 Methodology

### Pipeline Overview

1. **Input Data**

   * Clinical notes or social media posts

2. **PHE Classification (for tweets)**

   * Identify personal health experience content

3. **Symptom Extraction (BEL)**

   * Extract symptom mentions from raw text

4. **LLM-Based Normalization**

   * Convert informal expressions to preferred terms
   * Example:
     "can't breathe properly" → "Dyspnea"

5. **Embedding & Retrieval**

   * Generate SapBERT embeddings
   * Perform FAISS similarity search

6. **Concept Mapping**

   * Map to UMLS CUIs

---

## 📊 Results

### 1. Clinical Dataset (BCN – 3,144 phrases)

| Method                         | Accuracy  | F1 Score  |
| ------------------------------ | --------- | --------- |
| String Matching                | 0.679     | 0.809     |
| MetaMap Lite                   | 0.579     | 0.733     |
| Neural Embedding               | **0.858** | **0.924** |

---

### 2. Twitter Phrase-Level Dataset (BCN – 102 phrases)

| Method                         | Accuracy  | F1 Score  |
| ------------------------------ | --------- | --------- |
| String Matching                | 0.235     | 0.381     |
| MetaMap Lite                   | 0.118     | 0.211     |
| GPT-4o                         | 0.961     | 0.980 |
| GPT-4o mini                    | **0.980** | **0.990** |
| GPT-5                          | 0.941     | 0.970     |
| Gemini 2.0 flash               | 0.961     | 0.980     |
| Gemini 2.5 flash               | 0.882     | 0.937     |
| Meta Llama 3.1-70B             | 0.921     | 0.959     |
| Meta Llama 3.3-70B             | 0.941     | 0.970     |

---

### 3. Twitter Full-Text Dataset (BEL – 600 tweets)

#### Symptom Extraction Performance

| Metric Type | Exact Match F1 | Semantic Match F1 |
| ----------- | -------------- | ----------------- |
| Macro F1    | 0.896          | 0.918             |
| Micro F1    | 0.906          | 0.930             |

* Total symptom mentions extracted: 2125
* True positives: 1944

---

## 📈 Key Insights

* Embedding-based retrieval performs strongly on **clinical text**
* LLM-based normalization is essential for **informal social media data**
* Hybrid LLM + embedding approach provides **robust cross-domain performance**
* Enables end-to-end pipeline from raw text → structured biomedical concepts

---

## 🛠 Tech Stack

* Python
* HuggingFace Transformers (SapBERT)
* FAISS
* LangChain
* OpenAI / Gemini / Llama
* Pandas, NumPy, Matplotlib

---

## ▶️ How to Run

```bash
git clone https://github.com/iramazam3/biomedical-concept-normalization
cd biomedical-concept-normalization

pip install -r requirements.txt
python main.py
```

---

## 📷 Sample Output

Input:
"I feel feverish and can't taste anything."

Output:
Fever → C0015967
Ageusia → C2364111

"I feel [Fever] (C0015967) and [Ageusia] (C2364111)."

---

## 📦 Data Access

Due to licensing restrictions (e.g., UMLS), full datasets are not included.
Sample data is provided in the `/data/` directory.

---

## 📚 Citation

If you use this work, please cite:

> Azam, I., Jiang, K., & Bernard, G. (2026, March). Normalizing Health Concepts with Biomedical Embedding and LLMs. In Proceedings of the 1st Workshop on Linguistic Analysis for Health (HeaLing 2026) (pp. 180-190).

---

## 👤 Author

Iram Azam

Purdue University

---
