import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

# Page Config
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄"
)

st.title("📄 Resume Screening System")
st.write("Upload Resume Dataset & Paste Job Description")

# Upload CSV
uploaded_file = st.file_uploader("Upload Resume CSV", type=["csv"])

# Stopwords
stop_words = set(stopwords.words("english"))

# Skill List (You can add more)
SKILLS = [
    "python", "java", "sql", "c++", "c", "html", "css", "javascript",
    "machine learning", "deep learning", "nlp", "pandas", "numpy",
    "tensorflow", "keras", "power bi", "tableau", "excel", "aws",
    "docker", "git", "linux"
]

# Preprocess
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# Extract Name (basic)
def extract_name(text):
    lines = text.split("\n")

    for line in lines[:5]:
        line = line.strip()

        if len(line.split()) <= 3 and line.isalpha():
            return line.title()

    return "Not Found"


# Extract Skills
def extract_skills(text):

    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill.title())

    if len(found) == 0:
        return "Not Mentioned"

    return ", ".join(found)


# Main App
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded")

    if "Resume" not in df.columns or "Category" not in df.columns:
        st.error("CSV must contain Resume and Category columns")
        st.stop()

    # Add Resume ID
    df["Resume_ID"] = range(1, len(df) + 1)

    # Clean text
    df["cleaned_resume"] = df["Resume"].apply(preprocess)

    # Extract name & skills
    df["Name"] = df["Resume"].apply(extract_name)
    df["Skills"] = df["Resume"].apply(extract_skills)

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["cleaned_resume"])

    # Job Description
    st.subheader("📝 Job Description")

    job_description = st.text_area("Paste Job Description")

    if st.button("🔍 Match Resumes"):

        if job_description.strip() == "":
            st.warning("Enter Job Description")

        else:

            jd_cleaned = preprocess(job_description)

            jd_vector = tfidf.transform([jd_cleaned])

            similarity = cosine_similarity(jd_vector, tfidf_matrix)

            df["Match"] = similarity[0] * 100

            top_resumes = df.sort_values(
                "Match",
                ascending=False
            ).head(5)

            st.subheader("🏆 Top Candidates")

            for _, row in top_resumes.iterrows():

                st.markdown(f"### 🆔 Resume ID: {row['Resume_ID']}")


                st.write(f"📂 Category: {row['Category']}")
                st.write(f"🛠 Skills: {row['Skills']}")
                st.write(f"✅ Match: {round(row['Match'],2)} %")

                st.progress(min(row["Match"]/100, 1))

                st.markdown("---")

else:
    st.info("Upload dataset to start")