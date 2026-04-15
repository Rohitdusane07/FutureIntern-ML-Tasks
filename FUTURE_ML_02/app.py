import streamlit as st
import pandas as pd
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
from nltk.corpus import stopwords

# ---------------- Page Config ----------------
st.set_page_config(page_title="Ticket Classifier", page_icon="🎫")

# ---------------- Title ----------------
st.title("🎫 Support Ticket Classification System")
st.write("Classify support tickets and predict priority using Machine Learning")

# ---------------- Load Data ----------------
df = pd.read_csv("tickets.csv", encoding='latin1')

df = df[['Ticket Description', 'Ticket Type']]
df.rename(columns={
    'Ticket Description': 'text',
    'Ticket Type': 'category'
}, inplace=True)

# ---------------- Create Priority ----------------
def assign_priority(text):
    text = str(text).lower()
    if "urgent" in text or "immediately" in text or "not working" in text:
        return "High"
    elif "issue" in text or "problem" in text:
        return "Medium"
    else:
        return "Low"

df['priority'] = df['text'].apply(assign_priority)

# ---------------- Text Cleaning ----------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# ---------------- Model ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

cat_model = LogisticRegression(max_iter=200)
cat_model.fit(X, df['category'])

pri_model = LogisticRegression(max_iter=200)
pri_model.fit(X, df['priority'])

# ---------------- User Input ----------------
st.subheader("📝 Enter Support Ticket")

user_input = st.text_area("Type your issue here...", height=150)

# ---------------- Predict Button ----------------
if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a ticket description")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        category = cat_model.predict(vec)[0]
        priority = pri_model.predict(vec)[0]

        # ---------------- Output UI ----------------
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"📂 Category: {category}")

        with col2:
            if priority == "High":
                st.error(f"🔥 Priority: {priority}")
            elif priority == "Medium":
                st.warning(f"⚡ Priority: {priority}")
            else:
                st.info(f"✅ Priority: {priority}")