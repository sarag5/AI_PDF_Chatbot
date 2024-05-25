import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

def create_retrieval_system(documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.toarray().astype(np.float32))
    return vectorizer, index

def retrieve_documents(query, vectorizer, index, documents, k=5):
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]

model_name = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chatbot(pdf_path, query):
    pdf_text = extract_text_from_pdf(pdf_path)
    documents = pdf_text.split('\n')
    vectorizer, index = create_retrieval_system(documents)
    relevant_docs = retrieve_documents(query, vectorizer, index, documents)
    prompt = " ".join(relevant_docs) + "\n\nQ: " + query + "\nA:"
    response = generate_response(prompt)
    return response

if __name__ == "__main__":
    pdf_path = 'path_to_your_pdf.pdf'
    query = "What is the main topic of this document?"
    response = chatbot(pdf_path, query)
    print(response)
