import streamlit as st
import pickle
import re
import nltk
from io import StringIO
from PyPDF2 import PdfReader

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Loading models
with open('clf.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Define the resume cleaning function
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

# Define function to extract text from PDF files
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Mapping of prediction ID to category names
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Apply custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
        }
        .sidebar .sidebar-content {
            background-color: #005f73;
            color: white;
        }
        .stFileUploader {
            background-color: #e9d8a6;
            border: 2px dashed #005f73;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #005f73;
            color: white;
            border-radius: 5px;
        }
        .stWrite {
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSubtitle {
            color: #005f73;
        }
    </style>
""", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.title("Resume Screening App")
    st.write("Upload your resume in TXT or PDF format to get a category prediction.")

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        try:
            # Extract text based on file type
            if file_type == 'application/pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                # Assuming the file is a text file
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')

            # Clean the input resume
            cleaned_resume = clean_resume(resume_text)

            # Transform the cleaned resume using the trained TfidfVectorizer
            input_features = tfidf.transform([cleaned_resume])

            # Make the prediction using the loaded classifier
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_name = category_mapping.get(prediction_id, "Unknown")

            # Display the prediction
            st.subheader("Prediction")
            st.write(f"Resume Category: {category_name}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()