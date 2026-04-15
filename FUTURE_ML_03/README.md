# 📄 AI Resume Screening & Candidate Ranking System

## 🔍 Overview
This project is a Machine Learning-based Resume Screening System that analyzes resumes and matches them with job descriptions.

It helps recruiters quickly identify suitable candidates by calculating a match score, highlighting skills, and identifying gaps.

---

## 🎯 Objective
- Compare resumes with job descriptions  
- Calculate match score (%)  
- Identify matched and missing skills  
- Assist in candidate shortlisting  

---

## 🛠️ Technologies Used
- Python  
- NLTK  
- Scikit-learn  
- Sentence Transformers  
- Streamlit  

---

## ⚙️ Features

✔ Resume upload (PDF/DOCX)  
✔ Job description input  
✔ Semantic similarity using transformer models  
✔ Skill-based keyword matching  
✔ Match score calculation  
✔ Matched & missing skills detection  
✔ Interactive UI dashboard  

---

## 🧠 How It Works

### 1. Text Extraction
- Extracts text from resume (PDF/DOCX)

### 2. Text Preprocessing
- Lowercasing  
- Removing punctuation  
- Cleaning text  

### 3. Semantic Similarity
- Uses Sentence Transformers  
- Understands meaning of text  

### 4. Skill Matching
- Compares keywords between resume and job description  

### 5. Final Score
Final score is calculated as:

Final Score = (Semantic Score × Weight) + (Skill Match Score × Weight)

---

## 📊 Output

The system generates:
- Match Score (%)  
- Semantic similarity score  
- Skill match score  
- Matched skills  
- Missing skills  
- Suggestions for improvement  

---

## 💼 Business Use Case
This system helps:
- Recruiters shortlist candidates faster  
- Reduce manual resume screening  
- Identify skill gaps  
- Improve hiring efficiency  

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install streamlit pandas scikit-learn nltk PyPDF2 python-docx sentence-transformers