# DataScience
# 💼 Resume Screening AI (NLP-Based Ranking & Classification System)
📌 Project Overview

Resume Screening AI is an intelligent NLP-based system that helps recruiters automatically:

✅ Classify resumes into job categories

✅ Rank resumes based on job description similarity

✅ Generate match percentage scores

✅ Highlight matched keywords

✅ Export shortlisted candidates

The system uses Natural Language Processing (NLP) and Machine Learning techniques to match resumes with job requirements efficiently.

🎯 Problem Statement

Recruiters receive hundreds of resumes for a single job opening.
Manually reviewing them is:

Time-consuming

Error-prone

Inefficient

This project automates resume screening using text similarity and classification models to assist in smart shortlisting.

🚀 Key Features
🔹 Resume Ranking (Similarity-Based)

Upload job description

Upload multiple resumes (PDF)

Calculate cosine similarity

Generate match %

Rank candidates automatically

🔹 Resume Classification (ML-Based)

Predict job category:

Data Science

Software Engineering

Healthcare

Finance

HR

Engineering

and more...

🔹 Explainability

Highlight matched keywords

Display similarity score

Show predicted category

🔹 Web Application

Built using Streamlit

Clean and interactive UI

CSV download for shortlisted candidates

🛠️ Technologies Used
Category	Tools Used
Programming	Python
Data Handling	Pandas, NumPy
NLP	NLTK, Regex
Feature Engineering	TF-IDF
Machine Learning	Scikit-learn (SVM / Logistic Regression)
Similarity	Cosine Similarity
Web App	Streamlit
PDF Extraction	PyPDF2
📂 Dataset
Primary Dataset

File: resume_dataset.csv

169 resumes

Columns:

Category

Resume

Extended Dataset (Advanced Version)

2484 resumes

24+ job categories

Used for improved classification accuracy

⚙️ Project Execution Flow
🔹 Day 1 – Dataset Understanding

Loaded dataset using Pandas

Checked shape and structure

Explored category distribution

Displayed sample resumes

🔹 Day 2 – Text Preprocessing

Converted text to lowercase

Removed numbers and special characters

Removed stopwords

Handled encoding issues

Created cleaned_resume column

🔹 Day 3 – TF-IDF Vectorization

Applied TfidfVectorizer

Converted text into numerical feature vectors

Generated TF-IDF matrix

🔹 Day 4 – Similarity Calculation

Transformed job description using TF-IDF

Applied cosine similarity

Ranked resumes based on similarity score

🔹 Day 5 – Model Building

Trained classification model

Used:

LinearSVC / Logistic Regression

Evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🔹 Day 6 – Model Improvement

Hyperparameter tuning

Used larger dataset

Improved accuracy to ~74%

🔹 Day 7 – Deployment

Saved trained model (.pkl)

Built Streamlit application

Integrated:

Resume upload

Job description input

Ranking system

Category prediction

CSV export

📊 Model Performance (Improved Version)

Accuracy: ~74%

Multi-class classification (24 categories)

Balanced macro & weighted F1 score

🖥️ Application Workflow

Paste Job Description

Upload Resume PDFs

System:

Extracts text

Cleans text

Vectorizes text

Predicts category

Calculates similarity

Displays:

Ranked candidates

Match %

Matched keywords

Download shortlisted candidates as CSV

📸 System Architecture

Input → Preprocessing → TF-IDF →
Classification Model → Similarity Calculation → Ranking → Output

📊 Output

Ranked candidate list

Match percentage

Predicted job category

Highlighted keywords

Downloadable shortlist

📌 Future Improvements

Use BERT / Transformer embeddings

Improve subfield prediction (e.g., Data Science vs Software Engineering)

Add database integration

Deploy on cloud (Render / AWS / Streamlit Cloud)


---

