import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
import numpy as np

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
    threshold = 0.25
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

    # Initialize conversation chain
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # Initialize Sentence Transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # HR, IT, and Company Event keywords
    hr_it_event_keywords = [
    # HR Topics
    "Employee onboarding","name","who","are"
    "Exit interview",
    "Compensation policy",
    "Employee engagement programs",
    "Wellness programs",
    "Performance improvement plan",
    "Employee handbook",
    "Payroll management",
    "Recruitment policy",
    "Work-from-home policy",
    "Overtime policy",
    "Attendance and time-off tracking",
    "Employee conflict resolution",
    "Diversity and inclusion programs",
    "Health and safety policy",
    "Career development plans",
    "Training and development programs",
    "Succession planning",
    "Employee retention strategies",
    "Employee performance evaluations",
    "HR compliance and regulations",
    "Grievance handling",
    "Employee benefits (health insurance, pension, etc.)",
    "Employee discounts and perks",
    "Employment contract policies",
    "Leave of absence policies (maternity, paternity, medical, etc.)",
    "Employee recognition programs",
    "Code of conduct",
    "Disciplinary procedures",
    "Equal opportunity policies",
    "Workplace harassment and discrimination policies",
    "Travel and expense policies",
    "Relocation assistance",
    "Retirement planning and pensions",
    "Employment verification",
    "Job description and job analysis",

    # IT Support Topics
    "IT helpdesk ticketing system",
    "Software installation and updates",
    "Hardware troubleshooting (printers, monitors, laptops)",
    "Network configuration and issues",
    "Internet connectivity problems",
    "Email setup and configuration",
    "System login issues",
    "Software licensing management",
    "Data backup and recovery",
    "Antivirus and malware protection",
    "IT security policies",
    "VPN setup and troubleshooting",
    "Password reset and recovery",
    "Remote desktop access",
    "Application access permissions",
    "Cloud service integration",
    "Mobile device management",
    "Hardware procurement policies",
    "Technical training for employees",
    "Office phone system setup",
    "Internal software development tools support",
    "Cybersecurity awareness training",
    "Printer and peripheral device issues",
    "Server maintenance and monitoring",
    "IT disaster recovery plans",
    "Multi-factor authentication (MFA)",
    "Data storage and compliance",
    "Software compatibility issues",
    "Virtual machine setup and management",
    "Device encryption policies",
    "IT asset management",
    "Network security and firewall management",
    "Patch management",
    "Secure file sharing solutions",
    "IT vendor management",
    "Technical onboarding for new employees",

    # Company Events and Meetings
    "Annual general meeting (AGM)",
    "Holiday parties",
    "Company anniversary celebrations",
    "Team-building activities",
    "Corporate retreats",
    "Departmental meetings",
    "Town hall meetings",
    "Employee recognition events"
]

    # Encode the keywords for HR, IT, and Company Events
    category_embeddings = embedding_model.encode(hr_it_event_keywords, convert_to_tensor=True)

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
        user_question_ = "You are a Domain-Specific LLM named GailGPT, You are meant only to do this - (The chatbot should be capable of handling diverse questions related to HR policies, IT support, company events, and other organizational matters. (Hackathon students/teams to use publicly available sample information for HR Policy, IT Support, etc. available on internet.) Develop document processing capabilities for the chatbot to analyse and extract information from documents uploaded by employees. This includes summarizing a document or extracting text (keyword information) from documents relevant to organizational needs. (Hackathon students/teams can use any 8 to 10 page document for demonstration). Ensure the chatbot architecture is scalable to handle minimum 5 users parallelly. This includes optimizing response time (Response Time should not exceed 5 seconds for any query unless there is a technical issue like connectivity, etc.) Enable 2FA (2 Factor Authentication – email id type) in the chatbot for enhancing the security level of the chatbot. Chatbot should filter bad language as per system-maintained dictionary)"
        response = conversation(user_question_)
        ai_response = response.get("response", "No response text found")
        is_relevant_question(user_question_, embedding_model, category_embeddings)
        st.write("You are authenticated. Feel free to ask questions about HR policies, IT support, or company events!")

        # User input field for questions
        user_question = st.text_area("Ask your question:")

        if user_question:
            response = conversation(user_question)
            ai_response = response.get("response", "No response text found")  # Get the response text

            # Check if the question is relevant to HR, IT, or Company Events
            if is_relevant_question(user_question, embedding_model, category_embeddings):
                append_to_history(user_question, ai_response)
                with st.expander("Click to see AI's response"):
                    st.markdown(ai_response)
            else:
                st.write("Sorry, the response is not relevant to the topics covered (HR, IT, or Company Events). Please ask a related question.")

        # Show previous interactions
        if 'history' in st.session_state and st.session_state.history:
            if st.checkbox("Show Previous Interactions"):
                for idx, interaction in enumerate(reversed(st.session_state.history)):
                    with st.expander(f"Interaction {idx + 1}"):
                        st.markdown(f"*User Input:* {interaction['user_input']}")
                        st.markdown(f"*GAILGPT's Response:* {interaction['ai_response']}")
