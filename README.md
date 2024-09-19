# GailGPT: Intelligent Enterprise Assistant

GailGPT is an intelligent chatbot assistant designed to assist users with queries related to **HR**, **IT**, and **Company Events**. This application uses **LangChain**, **FAISS**, and **Sentence Transformers** to deliver relevant information and can handle document uploads for file-based question-answering. Additionally, the app incorporates an OTP-based email authentication system to secure access.

## Features

- **OTP Authentication**: Users must authenticate via OTP sent to their email before interacting with the assistant.
- **Question Answering**: The assistant is fine-tuned to respond to HR, IT, and Company Event-related questions using LangChain and FAISS.
- **Document Upload Support**: Users can upload PDF files, and the assistant will use the content of the documents to answer questions.
- **Conversational Memory**: The assistant remembers the last 5 user interactions for smoother conversations.

## Technologies Used

- **Streamlit**: For creating the web interface.
- **LangChain**: For managing conversational AI interactions and document-based querying.
- **Sentence Transformers**: For embedding text and similarity-based question relevance detection.
- **FAISS**: To handle vector search and retrieval from documents.
- **PyPDF2**: To read and process PDF documents.
- **smtplib** and **email.mime.text**: For sending OTP emails.
- **ChatGroq**: For large language model (LLM) conversations.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kabillanta/gailgpt.git
   cd gailgpt
   ```

2. **Set up the environment**:

   It's recommended to use a virtual environment. You can create one using `venv`:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   You can run the Streamlit app using the following command:

   ```bash
   streamlit run app.py
   ```

## Usage

1. **Email Authentication**: Enter your email to receive an OTP. You must validate this OTP to proceed.
2. **Ask Questions**: Once authenticated, you can ask questions related to HR, IT, and Company Events.
3. **Upload Documents**: Upload a PDF (or DOCX in the future) file for document-based question-answering. The assistant will process the document and use its contents to answer your questions.
4. **View Conversation History**: You can view past interactions in the "Show Previous Interactions" dropdown.

## Future Features

- **Support for DOCX Files**: Currently, the app supports PDF files. Support for DOCX files can be added by uncommenting the relevant code in the file uploader section.
- **Improved NLP Models**: Incorporating more powerful embeddings and retrieval models for more accurate responses.

## License

This project is licensed under the MIT License.
