import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (only first time)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# -----------------------------
# Text Cleaning Function
# -----------------------------
def preprocess(text):

    text = str(text).lower()

    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Resume Screening System")

st.title("📄 Resume Screening System")
st.write("Upload Resume Dataset & Paste Job Description")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Check columns
    if "Resume" not in df.columns or "Category" not in df.columns:
        st.error("CSV must contain 'Resume' and 'Category' columns")
        st.stop()

    # -----------------------------
    # Clean Resumes
    # -----------------------------
    df["cleaned_resume"] = df["Resume"].apply(preprocess)

    # -----------------------------
    # Job Description Input
    # -----------------------------
    st.write("### Paste Job Description")

    job_description = st.text_area("Enter Job Description Here")

    if st.button("Find Best Matches"):

        if job_description.strip() == "":
            st.warning("Please enter Job Description")
            st.stop()

        # Clean JD
        cleaned_jd = preprocess(job_description)

        # -----------------------------
        # TF-IDF Vector
        # -----------------------------
        tfidf = TfidfVectorizer()

        tfidf_matrix = tfidf.fit_transform(df["cleaned_resume"])

        jd_vector = tfidf.transform([cleaned_jd])

        # -----------------------------
        # Cosine Similarity
        # -----------------------------
        similarity = cosine_similarity(jd_vector, tfidf_matrix)[0]

        df["match_score"] = similarity * 100   # Percentage

        # -----------------------------
        # Sort Top Results
        # -----------------------------
        top_matches = df.sort_values(
            by="match_score",
            ascending=False
        ).head(5)

        st.write("## 🔝 Top Matching Resumes")

        # -----------------------------
        # Display Results
        # -----------------------------
        for i, row in top_matches.iterrows():

            st.markdown("---")

            st.write(f"📌 **Category:** {row['Category']}")
            st.write(f"✅ **Match:** {row['match_score']:.2f} %")

            preview = row["Resume"][:400] + "..."

            st.write("📝 Resume Preview:")
            st.info(preview)

else:
    st.info("👆 Upload a CSV file to begin")