# PDF Chatbot

This project implements a PDF chatbot using Retrieval-Augmented Generation (RAG) and fine-tuning with OpenAI's language models. The chatbot can extract information from PDF files and generate responses based on user queries.

## Features

- Extract text from PDF files
- Implement a simple retrieval system using TF-IDF and FAISS
- Integrate with OpenAI's BART language model
- Generate responses based on retrieved information

## Requirements

- Python 3.x
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [transformers](https://pypi.org/project/transformers/)
- [faiss-cpu](https://pypi.org/project/faiss-cpu/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sarag5/pdf-chatbot.git
    cd pdf-chatbot
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your PDF file in the project directory.

2. Modify the `main.py` file to point to your PDF file and set your query:
    ```python
    if __name__ == "__main__":
        pdf_path = 'path_to_your_pdf.pdf'  # Update with your PDF file path
        query = "What is the main topic of this document?"
        response = chatbot(pdf_path, query)
        print(response)
    ```

3. Run the script:
    ```bash
    python pdf_chatbot.py
    ```

## Code Overview

### `pdf_chatbot.py`

- `extract_text_from_pdf(pdf_path)`: Extracts text from the specified PDF file.
- `create_retrieval_system(documents)`: Creates a retrieval system using TF-IDF and FAISS.
- `retrieve_documents(query, vectorizer, index, documents, k=5)`: Retrieves the top `k` documents relevant to the query.
- `generate_response(prompt)`: Generates a response using OpenAI's BART language model.
- `chatbot(pdf_path, query)`: Combines all steps to generate a response based on the PDF content and user query.

## Example

```python
if __name__ == "__main__":
    pdf_path = 'example.pdf'  # Ensure this file exists in the directory
    query = "What is the main topic of this document?"
    response = chatbot(pdf_path, query)
    print(response)
```

## Contributing

Feel free to submit issues, fork the repository, and make pull requests. Any contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
