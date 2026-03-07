import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Resume Screening AI", page_icon="📄", layout="wide")

# -------- SMALL FONT STYLE --------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Resume Screening AI System")

st.markdown("""
AI-powered **Resume Ranking System** that evaluates resumes against a 
**Job Description using NLP and TF-IDF similarity**.
""")

# -------- STOPWORDS --------
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

stop_words = load_stopwords()

# -------- TEXT CLEANING --------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------- PDF TEXT EXTRACTION --------
def extract_pdf_text(file):

    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# -------- FILE UPLOAD --------
uploaded_files = st.file_uploader(
    "Upload Resumes (CSV or PDFs)",
    type=["csv","pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded")

job_description = st.text_area("Paste Job Description")

# -------- NUMBER OF CANDIDATES CONTROL --------
top_n = st.slider(
    "Select Number of Candidates to Shortlist",
    min_value=1,
    max_value=20,
    value=5
)

# -------- RUN SCREENING --------
if st.button("Run Screening"):

    if uploaded_files and job_description:

        resumes = []

        for file in uploaded_files:

            if file.name.endswith(".csv"):

                df_csv = pd.read_csv(file)

                if "Resume" not in df_csv.columns:
                    st.error("CSV must contain 'Resume' column")
                    st.stop()

                resumes.extend(df_csv["Resume"].astype(str).tolist())

            elif file.name.endswith(".pdf"):

                text = extract_pdf_text(file)

                if text.strip():
                    resumes.append(text)

        df = pd.DataFrame({"Resume": resumes})

        st.success(f"{len(df)} resumes loaded successfully")

        # -------- PREPROCESS --------
        df["cleaned_resume"] = df["Resume"].apply(preprocess)

        cleaned_jd = preprocess(job_description)

        tfidf = TfidfVectorizer(
            ngram_range=(1,2),
            stop_words="english"
        )

        tfidf_matrix = tfidf.fit_transform(df["cleaned_resume"])

        jd_vector = tfidf.transform([cleaned_jd])

        scores = cosine_similarity(jd_vector, tfidf_matrix)[0]

        df["match_percent"] = np.round(scores*100,2)

        df = df.sort_values(
            by="match_percent",
            ascending=False
        ).reset_index(drop=True)

        feature_names = tfidf.get_feature_names_out()

        jd_vector_array = jd_vector.toarray()[0]

        jd_keywords = [
            feature_names[i]
            for i in range(len(jd_vector_array))
            if jd_vector_array[i] > 0
        ]

        # -------- TABS --------
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Dashboard","🧠 Keywords","🏆 Top Candidates","📋 Shortlist"]
        )

        # -------- DASHBOARD --------
        with tab1:

            col1,col2,col3 = st.columns(3)

            col1.metric("Total Candidates", len(df))
            col2.metric("Highest Match %", f"{df['match_percent'].max()} %")
            col3.metric("Average Match %", round(df["match_percent"].mean(),2))

            st.subheader("Match Score Distribution")

            st.bar_chart(df["match_percent"])

        # -------- KEYWORDS --------
        with tab2:

            st.subheader("Extracted Job Keywords")

            st.write(", ".join(jd_keywords[:20]) if jd_keywords else "No keywords detected")

        # -------- TOP CANDIDATES --------
        with tab3:

            st.subheader("Top Ranked Candidates")

            top_candidates = df.head(top_n)

            for i,row in top_candidates.iterrows():

                with st.expander(f"Rank {i+1} | Match Score {row['match_percent']} %"):

                    resume_vector = tfidf.transform(
                        [row["cleaned_resume"]]
                    ).toarray()[0]

                    matched_keywords = [
                        feature_names[j]
                        for j in range(len(jd_vector_array))
                        if jd_vector_array[j] > 0 and resume_vector[j] > 0
                    ]

                    missing_skills = list(
                        set(jd_keywords) - set(matched_keywords)
                    )

                    st.write(
                        "Matched Skills:",
                        ", ".join(matched_keywords[:10]) if matched_keywords else "None"
                    )

                    st.write(
                        "Missing Skills:",
                        ", ".join(missing_skills[:5]) if missing_skills else "None"
                    )

                    st.write("Resume Preview:")

                    st.write(row["Resume"][:400] + "...")

        # -------- SHORTLIST --------
        with tab4:

            st.subheader("Shortlisted Candidates")

            top_candidates = df.head(top_n)

            st.dataframe(
                top_candidates[["Resume","match_percent"]]
            )

            csv = top_candidates.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Shortlisted Candidates",
                csv,
                "shortlisted_candidates.csv",
                "text/csv"
            )

    else:

        st.warning("Please upload resumes and paste job description.")