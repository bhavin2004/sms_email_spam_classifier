# 📩 SMS/Email Spam Classifier

This project is a **Streamlit-based SMS and Email Spam Classifier** that uses a machine learning pipeline to detect whether a given message is spam or ham.

---

## 🧠 **Model Details**
- Trained using **TF-IDF Vectorizer** and **Multinomial Naive Bayes Classifier**.
- Dataset: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Pipeline includes text preprocessing steps like tokenization, stemming, and vectorization.

---

## 💡 **Key Features**
✅ Predicts whether SMS/Email is spam or ham  
✅ Provides sample spam/ham messages for testing  
✅ Supports CSV uploads for bulk prediction  
✅ Download prediction results in CSV format  

---

## 🚀 **How to Run This Project**

### 1. Clone the Repository
```bash
git clone https://github.com/bhavin2004/sms_email_spam_classifier.git
cd sms_email_spam_classifier
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📤 **Using CSV for Bulk Prediction**
- Navigate to the `Predict with CSV` section.
- Upload a CSV file with a column named `v2` containing the messages.
- View and download the prediction results.

---

## 📦 **Folder Structure**
```
/sms_email_spam_classifier
├── /artifacts
│   ├── spam.svg
│   └── test_data.csv
├── /src
│   └── /sms_spam_classifier
│       ├── /pipelines
│       │   ├── prediction_pipeline.py
│       │   └── training_pipeline.py
│       └── /utils
│           └── some_utils.py
├── app.py
└── requirements.txt
```

---

## 🎨 **Technologies Used**
- Python 🐍
- Scikit-learn 🤖
- Pandas 🧮
- Streamlit 🎈

---

## 👨‍💻 **Developed by:**
- **Bhavin Karangia** 😎
- Aspiring Data Scientist with expertise in ML, AI, and Web Scraping.

---

## 📢 **License**
This project is open source and available under the [MIT License](LICENSE).

