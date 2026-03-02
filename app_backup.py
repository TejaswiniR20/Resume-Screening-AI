import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (first time only)
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -------------------------------
# Text Cleaning Function
# -------------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Resume Screening System")

st.title("📄 Resume Screening System")
st.write("Upload Resume Dataset & Paste Job Description")

# Upload CSV
uploaded_file = st.file_uploader("Upload Resume CSV", type="csv")

# Job Description
job_description = st.text_area("Paste Job Description Here")

# -------------------------------
# Process
# -------------------------------
if uploaded_file and job_description:

    df = pd.read_csv(uploaded_file)

    if "Resume" not in df.columns:
        st.error("CSV must contain 'Resume' column")
        st.stop()

    st.success("Dataset Loaded Successfully!")

    # Clean resumes
    df["cleaned_resume"] = df["Resume"].apply(preprocess)

    # Clean JD
    cleaned_jd = preprocess(job_description)

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["cleaned_resume"])
    jd_vector = tfidf.transform([cleaned_jd])

    # Similarity
    scores = cosine_similarity(jd_vector, tfidf_matrix)[0]
    df["match_percent"] = (scores * 100).round(2)

    # Sort
    df = df.sort_values(by="match_percent", ascending=False)

    st.subheader("✅ Top Matching Resumes")

    top_n = st.slider("Select Top N Results", 1, 20, 5)

    for i in range(top_n):

        row = df.iloc[i]

        st.markdown("---")

        st.write(f"### 🧾 Resume #{i+1}")

        if "Category" in df.columns:
            st.write("📌 Category:", row["Category"])

        st.write("🎯 Match:", f"{row['match_percent']} %")

        st.write("📄 Resume Preview:")
        st.write(row["Resume"][:600] + "...")

else:
    st.info("Please upload a CSV file and enter Job Description.")