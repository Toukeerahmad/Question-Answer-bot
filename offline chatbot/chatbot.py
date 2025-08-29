import pdfplumber
import re
from transformers import pipeline

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Preprocess the Text
def preprocess_text(text):
    # Split text into sentences or paragraphs
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# Step 3: Set Up a Question-Answering Model
def setup_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 4: Create a QA Function
def answer_question(qa_model, question, context):
    result = qa_model(question=question, context=context)
    return result['answer']

# Step 5: Main Application Logic
def main():
    # Provide the path to your PDF
    pdf_path = "virat_kohli.pdf"  # Replace with your actual PDF file path
    
    # Extract text from the PDF
    print("Extracting text from the PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    # Preprocess the text
    print("Preprocessing text...")
    context = " ".join(preprocess_text(text))  # You can chunk text for large files

    # Load the QA model
    print("Loading the Question-Answering model...")
    qa_model = setup_qa_model()

    # Interactive Q&A Loop
    print("\nThe application is ready! Ask your questions:")
    while True:
        question = input("\nYour Question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        
        # Get the answer
        answer = answer_question(qa_model, question, context)
        print("Answer:", answer)

if __name__ == "__main__":
    main()