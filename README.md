# 🛡️ PolicyGuard NLP

**Automatic Detection of Policy Violations in Social Media Content**

---

## 📌 Overview

PolicyGuard NLP is an AI-powered content moderation system that detects harmful and policy-violating text in social media posts.
It uses a **BERT-based transformer model** to classify content into multiple toxicity categories and provides a **severity score** for better moderation decisions.

---

## 🚀 Features

* 🔍 **Multi-label Toxic Detection**
  Detects multiple categories like toxic, insult, threat, etc.

* 🤖 **BERT-based NLP Model**
  Uses `unitary/toxic-bert` for high accuracy.

* 📊 **Severity Scoring System**
  Assigns a severity level (Safe, Moderate, High, Critical).

* 🖥️ **Interactive Dashboard**
  Built with Streamlit for real-time content analysis.

* 📈 **Visualization**
  Displays category-wise probabilities using charts.

* 📚 **History Tracking**
  Stores previously analyzed inputs.

---

## 🧠 Model Details

* **Model**: `unitary/toxic-bert`
* **Framework**: HuggingFace Transformers
* **Backend**: PyTorch

### Categories Detected:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

---

## 📂 Project Structure

```
policy-guard-nlp/
│
├── app.py              # Streamlit dashboard
├── predict.py          # Prediction logic
├── requirements.txt    # Dependencies
├── README.md
│
├── dataset/            # (Optional)
├── models/             # (Optional)
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/ArjunKS474/policy-guard-nlp.git
cd policy-guard-nlp
```

2. Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📊 How It Works

1. User enters text
2. Text is tokenized using BERT tokenizer
3. Model predicts probabilities for each category
4. Severity score is calculated
5. Results displayed on dashboard

---

## 🔮 Future Improvements

* Context-aware toxic detection
* Explainable AI (LIME/SHAP)
* FastAPI integration
* Real-time moderation alerts
* Deployment on cloud

---

## 📸 Demo

*Add your screenshots here*

---

## ⭐ Support

If you like this project, please ⭐ the repository!
