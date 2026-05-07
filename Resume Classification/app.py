import streamlit as st
import pickle
import pandas as pd
import re
import zipfile
import io
from PyPDF2 import PdfReader
from docx import Document

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# -------------------------------
# HEADER
# -------------------------------
st.title("🚀 AI Resume Screening Dashboard")
st.write("Upload resumes (PDF, DOCX, or ZIP) and rank candidates automatically")

# -------------------------------
# SKILLS DATABASE
# -------------------------------
skills_list = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "java",
    "react",
    "javascript",
    "html",
    "css",
    "django",
    "flask",
    "power bi",
    "tableau"
]

# -------------------------------
# SIDEBAR JOB CONFIG
# -------------------------------
st.sidebar.title("⚙ Job Configuration")

job_input = st.sidebar.text_input(
    "Enter Required Skills",
    "python, sql, machine learning"
)

job_skills = [skill.strip().lower() for skill in job_input.split(",")]

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Resumes or ZIP Folder",
    type=["pdf","docx","zip"],
    accept_multiple_files=True
)

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text

# -------------------------------
# EXTRACT TEXT
# -------------------------------
def extract_text(file):

    text = ""

    try:

        if hasattr(file, "type") and file.type == "application/pdf":

            reader = PdfReader(file)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

        else:

            doc = Document(file)

            for para in doc.paragraphs:

                text += para.text + " "

    except:
        pass

    return text

# -------------------------------
# SKILL EXTRACTION
# -------------------------------
def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_list:

        if skill in text:
            found_skills.append(skill)

    return found_skills

# -------------------------------
# MATCH CALCULATION
# -------------------------------
def calculate_match(resume_skills, job_skills):

    if len(job_skills) == 0:
        return 0

    match = len(set(resume_skills) & set(job_skills))

    return round((match/len(job_skills))*100,2)

# -------------------------------
# MAIN PROCESS
# -------------------------------
if uploaded_files:

    files_to_process = []

    for file in uploaded_files:

        # ZIP FILE
        if file.name.endswith(".zip"):

            zip_file = zipfile.ZipFile(io.BytesIO(file.read()))

            for name in zip_file.namelist():

                if name.endswith((".pdf",".docx")):

                    files_to_process.append((name, zip_file.open(name)))

        else:

            files_to_process.append((file.name, file))

    results = []

    for name, file in files_to_process:

        text = extract_text(file)

        if not text:
            continue

        cleaned = clean_text(text)

        vec = vectorizer.transform([cleaned])

        role = model.predict(vec)[0]

        skills = extract_skills(cleaned)

        match = calculate_match(skills, job_skills)

        results.append({

            "Resume Name": name,
            "Predicted Role": role,
            "Skills": " | ".join(skills),
            "Match %": match

        })

    df = pd.DataFrame(results)

    if not df.empty:

        df = df.sort_values(by="Match %", ascending=False)

        # ---------------- METRICS ----------------
        col1,col2,col3 = st.columns(3)

        col1.metric("📄 Total Resumes",len(df))
        col2.metric("🏆 Highest Match %",f"{df['Match %'].max()}%")
        col3.metric("📊 Average Match %",f"{round(df['Match %'].mean(),2)}%")

        # ---------------- TABLE ----------------
        st.subheader("📋 Resume Ranking")

        st.dataframe(df)

        # ---------------- TOP 10 ----------------
        st.subheader("🏆 Top Candidates")

        st.dataframe(df.head(10))

        # ---------------- FILTER ----------------
        selected_skill = st.selectbox(
            "Filter by Skill",
            ["All"] + skills_list
        )

        if selected_skill != "All":

            filtered_df = df[df["Skills"].str.contains(selected_skill, case=False, na=False)]

            st.dataframe(filtered_df)

        # ---------------- CHART ----------------
        st.subheader("📊 Candidate Ranking Chart")

        st.bar_chart(df.set_index("Resume Name")["Match %"])

        # ---------------- DOWNLOAD ----------------
        st.download_button(
            label="Download Ranking CSV",
            data=df.to_csv(index=False),
            file_name="resume_ranking.csv",
            mime="text/csv"
        )