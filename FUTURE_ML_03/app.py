import streamlit as st
import nltk
import string

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from PyPDF2 import PdfReader
import docx

nltk.download('stopwords')
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------- FILE READERS ----------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = text.replace(",", " ")
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ---------------- UI ----------------
st.title("📄 AI Resume Screening & Ranking System")
st.markdown("### Match your resume with job roles using AI 🚀")

# Sidebar
st.sidebar.header("⚙️ Settings")
weight_semantic = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.7)
weight_skill = 1 - weight_semantic

# Input section
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload Resume", type=["pdf", "docx"])

with col2:
    job_desc = st.text_area("📝 Paste Job Description", height=200)

# ---------------- PROCESS ----------------
if st.button("🔍 Analyze Resume"):

    if uploaded_file is None or job_desc.strip() == "":
        st.warning("⚠️ Please upload resume and enter job description")
    else:
        # Read file
        if uploaded_file.type == "application/pdf":
            resume_text = read_pdf(uploaded_file)
        else:
            resume_text = read_docx(uploaded_file)

        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_desc)

        # -------- SEMANTIC SIMILARITY --------
        resume_emb = model.encode([resume_clean])
        job_emb = model.encode([job_clean])

        semantic_score = cosine_similarity(resume_emb, job_emb)[0][0]

        # -------- SKILL MATCHING --------
        job_words = set(job_clean.split())
        resume_words = set(resume_clean.split())

        matched = job_words.intersection(resume_words)
        missing = job_words - resume_words

        skill_score = len(matched) / len(job_words) if len(job_words) > 0 else 0

        # -------- FINAL SCORE --------
        final_score = (weight_semantic * semantic_score) + (weight_skill * skill_score)

        # ---------------- OUTPUT UI ----------------
        st.markdown("---")
        st.subheader("📊 Analysis Result")

        # Score section
        score_percent = round(final_score * 100, 2)

        st.progress(int(score_percent))

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Match Score", f"{score_percent}%")

        with colB:
            st.metric("Semantic Score", f"{round(semantic_score*100,2)}%")

        with colC:
            st.metric("Skill Match", f"{round(skill_score*100,2)}%")

        # Status
        if final_score > 0.75:
            st.success("🔥 Excellent Match")
        elif final_score > 0.5:
            st.warning("⚡ Moderate Match")
        else:
            st.error("❌ Low Match")

        # -------- SKILLS --------
        st.markdown("### ✅ Matched Skills")
        if matched:
            st.success(", ".join(list(matched)[:20]))
        else:
            st.write("No strong matches")

        st.markdown("### ❗ Missing Skills")
        if missing:
            st.error(", ".join(list(missing)[:20]))
        else:
            st.success("No major skill gaps 🎉")

        # -------- SUGGESTIONS --------
        st.markdown("### 💡 Suggestions")
        st.info("Add missing skills and improve keywords to increase your match score.")

        # -------- SUMMARY --------
        st.markdown("### 📌 Summary")
        st.write(f"""
        This resume matches **{score_percent}%** with the given job description.
        Improving the missing skills can significantly increase chances of selection.
        """)