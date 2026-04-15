import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download stopwords (only first time)
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("tickets.csv", encoding='latin1')

# -----------------------------
# 2. Select & Rename Columns
# -----------------------------
df = df[['Ticket Description', 'Ticket Type']]

df.rename(columns={
    'Ticket Description': 'text',
    'Ticket Type': 'category'
}, inplace=True)

df.dropna(inplace=True)

# -----------------------------
# 3. Create Priority (IMPORTANT)
# -----------------------------
def assign_priority(text):
    text = str(text).lower()
    if "urgent" in text or "immediately" in text or "not working" in text:
        return "High"
    elif "issue" in text or "problem" in text:
        return "Medium"
    else:
        return "Low"

df['priority'] = df['text'].apply(assign_priority)

# -----------------------------
# 4. Text Cleaning
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# -----------------------------
# 5. Convert Text → Numbers (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

# Targets
y_category = df['category']
y_priority = df['priority']

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_cat_train, y_cat_test = train_test_split(X, y_category, test_size=0.2, random_state=42)

_, _, y_pri_train, y_pri_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)

# -----------------------------
# 7. Train Models
# -----------------------------
cat_model = LogisticRegression(max_iter=200)
cat_model.fit(X_train, y_cat_train)

pri_model = LogisticRegression(max_iter=200)
pri_model.fit(X_train, y_pri_train)

# -----------------------------
# 8. Predictions
# -----------------------------
cat_pred = cat_model.predict(X_test)
pri_pred = pri_model.predict(X_test)

# -----------------------------
# 9. Evaluation
# -----------------------------
print("Category Classification Report:\n")
print(classification_report(y_cat_test, cat_pred))

print("\nPriority Classification Report:\n")
print(classification_report(y_pri_test, pri_pred))

# -----------------------------
# 10. Test with custom input
# -----------------------------
sample = ["My account is not working and I need help urgently"]

sample_clean = [clean_text(sample[0])]
sample_vec = vectorizer.transform(sample_clean)

print("\nSample Prediction:")
print("Category:", cat_model.predict(sample_vec)[0])
print("Priority:", pri_model.predict(sample_vec)[0])