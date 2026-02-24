# DataScience
# Resume Screening AI

## 📌 Project Overview
This project implements a Resume Screening AI system that automatically ranks resumes based on their similarity to a given job description using Natural Language Processing (NLP) techniques.

The goal is to help recruiters shortlist candidates efficiently by matching resumes with job requirements.

---

## 🚀 Features
- Resume dataset loading and analysis
- Text preprocessing (cleaning and normalization)
- TF-IDF vectorization
- Cosine similarity-based resume ranking
- Similarity score generation

---

## 🛠️ Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Regular Expressions
- Google Colab / Jupyter Notebook

---

## 📂 Dataset
The dataset contains resumes labeled with job categories.

File used:
- `resume_dataset.csv`

Dataset Details:
- 169 resumes
- 2 columns: Category, Resume

---

## ⚙️ Project Execution Process

### 🔹 Day 1 – Dataset Understanding
- Loaded dataset using Pandas
- Checked dataset shape and structure
- Explored job categories
- Displayed sample resumes

### 🔹 Day 2 – Text Preprocessing
- Converted text to lowercase
- Removed numbers and special characters
- Removed stopwords
- Fixed encoding issues
- Created `cleaned_resume` column

### 🔹 Day 3 – TF-IDF Vectorization
- Applied `TfidfVectorizer`
- Converted cleaned resumes into numerical feature vectors
- Generated TF-IDF matrix

### 🔹 Day 4 – Resume Ranking
- Created a sample job description
- Transformed job description using TF-IDF
- Calculated cosine similarity
- Ranked resumes based on similarity score

---

## 📊 Output
- TF-IDF feature matrix
- Similarity score for each resume
- Ranked list of resumes matching job description

---

## 📌 Future Improvements
- Add PDF/DOCX resume upload feature
- Implement classification model
- Deploy as web application (Streamlit)
- Add keyword highlighting for explainability

---

