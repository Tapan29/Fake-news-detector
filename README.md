#Fake News Detector using Machine Learning

A machine learning project that classifies news articles as **REAL** or **FAKE** using NLP techniques like TF-IDF Vectorization and classification algorithms.

---

##Objective

With the rise of misinformation, detecting fake news has become critical. This project builds an ML pipeline to automatically classify news articles based on their content.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| NLP | Scikit-learn (TF-IDF) |
| ML Models | Logistic Regression, Naive Bayes |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Project Structure

```
fake-news-detector/
│
├── fake_news_detector.ipynb   # Main notebook
├── train.csv                  # Dataset (download from Kaggle)
├── README.md                  # Project documentation
└── images/
    ├── label_distribution.png
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

## Dataset

**Kaggle Fake News Dataset**  
🔗 https://www.kaggle.com/c/fake-news/data

- ~20,000 news articles
- Labels: `0 = REAL`, `1 = FAKE`
- Features: `title`, `author`, `text`

---

##  Workflow

```
Raw Data → Data Cleaning → TF-IDF Vectorization → Model Training → Evaluation → Prediction
```

1. **Data Preprocessing** — Lowercasing, removing URLs, punctuation, numbers
2. **Feature Engineering** — Combined title + text, TF-IDF with bigrams
3. **Model Training** — Logistic Regression & Naive Bayes
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
5. **Custom Prediction** — Predict any news article in real-time

---

##  Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~98% |
| Naive Bayes | ~95% |

---

##  How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Tapan29/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Download dataset from Kaggle and place train.csv in this folder

# 4. Launch Jupyter Notebook
jupyter notebook fake_news_detector.ipynb
```

---

## Future Improvements

- Deploy as a Streamlit web app
- Use BERT/transformer models for higher accuracy
- Add multilingual support
- Real-time news URL scanning

---

##  Author

**Tapan Singh Naruka**  
B.Tech IT | Acropolis Institute of Technology and Research, Indore  
narukatapan@gmail.com  
