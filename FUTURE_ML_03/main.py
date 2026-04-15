import pandas as pd
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# 1. Sample Data (you can replace later with dataset)
# -----------------------------
resumes = [
    "Python developer with machine learning and SQL skills",
    "Web developer skilled in HTML CSS JavaScript",
    "Data analyst with Python SQL and Excel experience"
]

job_description = "Looking for a candidate with Python Machine Learning and SQL skills"

# -----------------------------
# 2. Text Cleaning
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

resumes_clean = [clean_text(r) for r in resumes]
job_clean = clean_text(job_description)

# -----------------------------
# 3. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(resumes_clean + [job_clean])

# -----------------------------
# 4. Similarity Calculation
# -----------------------------
similarity = cosine_similarity(vectors[-1], vectors[:-1])

# -----------------------------
# 5. Ranking
# -----------------------------
scores = similarity[0]

for i, score in enumerate(scores):
    print(f"Resume {i+1} Match Score: {round(score*100, 2)}%")

# -----------------------------
# 6. Find Missing Skills
# -----------------------------
job_words = set(job_clean.split())

for i, resume in enumerate(resumes_clean):
    resume_words = set(resume.split())
    missing = job_words - resume_words
    print(f"\nResume {i+1} Missing Skills:", missing)