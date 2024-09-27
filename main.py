import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import numpy as np
import PyPDF2
from io import BytesIO

# Imports for RAG and FAISS-CPU
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Initialize Streamlit app
st.title("GailGPT\nIntelligent Enterprise Assistant")

# Function to send OTP via email
def send_otp(email):
    otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
    st.session_state.otp = otp  # Store OTP in session state

    # Email configuration
    sender_email = "gailgpt7@gmail.com"  # Replace with your email
    sender_password = "vkja kivi knbz uiiv"  # Replace with your app-specific password
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Compose email
    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

    try:
        # Connect to the SMTP server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        st.success(f"OTP sent to {email}")
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")

# Function to validate the OTP
def validate_otp(user_otp):
    if int(user_otp) == st.session_state.otp:
        st.session_state.authenticated = True
        st.success("OTP validated successfully!")
        st.rerun()  # Force a rerun to update the app state
    else:
        st.error("Invalid OTP. Please try again.")

# Function to check if the user's question is related to HR, IT, or Company Events
def is_relevant_question(question, embedding_model, category_embeddings):
    threshold = 0.01
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, category_embeddings)
    
    # Convert similarities tensor to a NumPy array
    similarities_np = similarities.numpy()

    # Find the maximum similarity using np.argmax and retrieve the value
    max_index = np.argmax(similarities_np)
    max_similarity = similarities_np.flatten()[max_index]

    return max_similarity > threshold

# Function to append conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# Function to read PDF files
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to filter responses
def filter_response(response):
    # Implement bad language filtering and other criteria here
    # For now, just return the original response
    return response

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=5)

# Groq API key
groq_api_key = "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"

if groq_api_key:
    # Initialize Groq Langchain chat object
    model_name = "llama3-70b-8192"
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name
    )

    # Define a custom prompt template
    template = """You are GailGPT, an intelligent enterprise assistant. Your primary functions are:
    1. Handle diverse questions related to HR policies, IT support, company events, and other organizational matters.
    2. Analyze and extract information from uploaded documents, including summarizing or extracting relevant keywords.
    3. Ensure responses are timely and relevant.
    4. Use 2-factor authentication for enhanced security.
    5. Filter out bad language as per the system-maintained dictionary.
    6. Stay focused on organizational topics and avoid engaging in unrelated tasks
    7. You have RAG capability (at least act like you have)

    Current conversation:
    {history}
    Human: {input}
    AI Assistant: """

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    # Initialize conversation chain with the custom prompt
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory,
        prompt=prompt
    )

    # Initialize Sentence Transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # HR, IT, and Company Event keywords
    hr_it_event_keywords = [
        # HR Topics
        "Employee onboarding", "name", "who", "are",
        "Exit interview",
        "Compensation policy",
        "Performance review",
        "Training and development",
        "Leave policy",
        "Employee benefits",
        "Workplace safety",
        "Diversity and inclusion",
        "Recruitment process",
        "Employee relations",
        
        # IT Topics
        "Password reset",
        "Software installation",
        "Network connectivity",
        "Email setup",
        "Hardware troubleshooting",
        "VPN access",
        "Data backup",
        "Cybersecurity",
        "Software licensing",
        "IT support ticket",
        
        # Company Events
        "Team building",
        "Annual meeting",
        "Holiday party",
        "Company picnic",
        "Charity event",
        "Product launch",
        "Awards ceremony",
        "Training workshop",
        "Retirement celebration",
        "Company anniversary"
    ]

    # Encode the keywords for HR, IT, and Company Events
    category_embeddings = embedding_model.encode(hr_it_event_keywords, convert_to_tensor=True)

    # Initialize RAG components
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize FAISS-CPU
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = FAISS.from_texts([""], embeddings)

    # OTP Authentication
    if not st.session_state.get('authenticated', False):
        # Ask user to enter email for OTP
        email = st.text_input("Enter your email to receive an OTP:")
        if st.button("Send OTP") and email:
            send_otp(email)

        # OTP Input field
        otp_input = st.text_input("Enter the OTP sent to your email:", type="password")
        if st.button("Validate OTP") and otp_input:
            validate_otp(otp_input)
    else:
        # Display chatbot functionality after authentication
        st.write("You are authenticated. Feel free to ask questions about HR policies, IT support, or company events!")

        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
        if uploaded_file is not None:
            # Read the file
            if uploaded_file.type == "application/pdf":
                file_content = read_pdf(uploaded_file)
            # Add handling for DOCX files if needed
            
            # Process the file content
            chunks = text_splitter.split_text(file_content)
            
            # Update the FAISS index with new documents
            new_vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.vectorstore.merge_from(new_vectorstore)
            
            st.success("File uploaded and processed successfully!")

        # Tabs for different query types
        tab1, tab2 = st.tabs(["General Questions", "Document-based Queries"])

        with tab1:
            st.subheader("Ask a general question")
            user_question = st.text_area("Your question:")
            if user_question:
                if is_relevant_question(user_question, embedding_model, category_embeddings):
                    response = conversation.predict(input=user_question)
                    filtered_response = filter_response(response)
                    append_to_history(user_question, filtered_response)
                    st.text_area("GailGPT's response:", value=filtered_response, height=200, disabled=True)
                else:
                    st.write("Sorry, the question is not relevant to the topics covered (HR, IT, or Company Events). Please ask a related question.")

        with tab2:
            st.subheader("Ask about uploaded documents")
            rag_question = st.text_area("Your document-related question:")
            if rag_question:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                rag_response = qa_chain(rag_question)
                st.text_area("Document-based answer:", value=rag_response['result'], height=200, disabled=True)

        # Show previous interactions in an expander (dropdown)
        if 'history' in st.session_state and st.session_state.history:
            with st.expander("Show Previous Interactions"):
                for idx, interaction in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**User:** {interaction['user_input']}")
                    st.markdown(f"**GailGPT:** {interaction['ai_response']}")
                    if idx < len(st.session_state.history) - 1:
                        st.markdown("---")
