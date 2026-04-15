# 🎫 Support Ticket Classification & Prioritization

## 🔍 Overview
This project focuses on building a Machine Learning system that automatically classifies customer support tickets and assigns priority levels.

It helps businesses reduce manual effort and improve response time by intelligently handling support requests.

---

## 🎯 Objective
- Classify support tickets into categories  
- Assign priority levels (High / Medium / Low)  
- Improve support efficiency using Machine Learning  

---

## 🛠️ Tools & Technologies
- Python  
- NLTK  
- Scikit-learn  
- Streamlit  
- VS Code  

---

## 📁 Dataset
The dataset contains support ticket details such as:
- Ticket Description  
- Ticket Type  

Priority is generated using rule-based logic.

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Removed missing values  
- Cleaned text (lowercase, punctuation removal)  
- Removed stopwords  

### 2. Feature Engineering
- Converted text into numerical form using TF-IDF  

### 3. Model Building
- Used Logistic Regression for classification  
- Built separate models for:
  - Ticket category  
  - Ticket priority  

### 4. Priority Logic
- High → urgent keywords  
- Medium → issue/problem keywords  
- Low → general queries  

### 5. Evaluation
- Used classification report (accuracy, precision, recall)  

---

## 📊 Features
- Automatic ticket classification  
- Priority prediction  
- Real-time prediction using Streamlit UI  

---

## 💼 Business Use Case
This system helps:
- Reduce manual ticket sorting  
- Prioritize urgent issues  
- Improve response time  
- Enhance customer satisfaction  

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn nltk streamlit