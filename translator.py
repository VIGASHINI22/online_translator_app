import os
import streamlit as st
import base64
from PyPDF2 import PdfReader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from docx import Document

# Create "temp" directory to store temporary files
os.makedirs("temp", exist_ok=True)

# Streamlit code
st.set_page_config(layout='wide', page_title="Document Translator")

@st.cache_data
def display_pdf(file):
    file_path = None

    if hasattr(file, "name"):  # Check if file is an UploadedFile
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.pdf':
            # Save the uploaded PDF file to a temporary location
            file_path = f"temp/{file.name}"
            with open(file_path, 'wb') as temp_file:
                temp_file.write(file.read())
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return

    if file_path:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_text_from_word(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def translate_arabic_to_english(article_ar):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "ar_AR"
    encoded_ar = tokenizer(article_ar, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    trans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return trans[0]

def main():
    st.title('Document Translator')
    uploaded_file = st.file_uploader("Upload your File", type=['pdf', 'txt', 'docx'])

    if uploaded_file is not None:
        st.info("File Uploaded Successfully!")

        # Display uploaded file
        col1, col2 = st.columns(2)
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == '.pdf':
            text_from_file = extract_text_from_pdf(uploaded_file)
            with col1:
                st.info("Uploaded PDF File")
                display_pdf(uploaded_file)

        elif file_extension == '.txt':
            text_from_file = extract_text_from_txt(uploaded_file)
            with col1:
                st.info("Uploaded Text File")
                st.text(text_from_file)

        elif file_extension == '.docx':
            text_from_file = extract_text_from_word(uploaded_file)
            with col1:
                st.info("Uploaded Word File")
                st.text(text_from_file)

        else:
            st.error(f"Unsupported file format: {file_extension}")
            return

        # Translate Arabic text to English
        translated_text = translate_arabic_to_english(text_from_file)

        # Display translated text
        with col2:
            st.info("Translated Text")
            st.text(translated_text)

if __name__ == "__main__":
    main()

