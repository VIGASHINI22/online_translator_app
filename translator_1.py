#uploading pdf and translating in
import os
from PyPDF2 import PdfReader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from docx import Document

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

if __name__ == "__main__":
    # Replace 'your_file_path' with the path to your file (PDF, TXT, or Word)
    file_path = '/content/1.pdf'

    # Extract text from the file based on its format
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        text_from_file = extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        text_from_file = extract_text_from_txt(file_path)
    elif file_extension == '.docx':
        text_from_file = extract_text_from_word(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Translate Arabic text to English
    translated_text = translate_arabic_to_english(text_from_file)

    print("\nTranslation:")
    print(translated_text)
