import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Function to initialize chatbot and model-related components
def initialize_chatbot():
    groq_api_key = "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"
    model_name = "llama3-70b-8192"
    
    if groq_api_key:
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name
        )
        conversation = ConversationChain(
            llm=groq_chat,
            memory=ConversationBufferWindowMemory(k=5)
        )
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return conversation, embedding_model
    return None, None

# Function to send OTP via email
def send_otp(email):
    otp = random.randint(100000, 999999)
    st.session_state.otp = otp

    sender_email = "gailgpt7@gmail.com"
    sender_password = "vkja kivi knbz uiiv"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        st.success(f"OTP sent to {email}")
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")

# Function to validate OTP
def validate_otp(user_otp):
    if 'otp' in st.session_state and int(user_otp) == st.session_state.otp:
        st.session_state.authenticated = True
        st.success("OTP validated successfully!")
        st.experimental_rerun()
    else:
        st.error("Invalid OTP. Please try again.")

# Function to check if the user's question is relevant
def is_relevant_question(question, embedding_model, category_embeddings, threshold=0.25):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, category_embeddings).numpy()
    max_similarity = np.max(similarities)
    return max_similarity > threshold

# Function to append conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# Function to handle user authentication
def authenticate_user():
    email = st.text_input("Enter your email to receive an OTP:")
    if st.button("Send OTP") and email:
        send_otp(email)

    otp_input = st.text_input("Enter the OTP sent to your email:", type="password")
    if st.button("Validate OTP") and otp_input:
        validate_otp(otp_input)

# Main function to handle chatbot interface
def chatbot_interface(conversation, embedding_model, category_embeddings):
    st.write("You are authenticated. Feel free to ask questions about HR policies, IT support, or company events!")
    
    user_question = st.text_area("Ask your question:")
    if user_question:
        response = conversation(user_question)
        ai_response = response.get("response", "No response text found")

        if is_relevant_question(user_question, embedding_model, category_embeddings):
            append_to_history(user_question, ai_response)
            with st.expander("Click to see AI's response"):
                st.markdown(ai_response)
        else:
            st.write("Sorry, the response is not relevant to the topics covered (HR, IT, or Company Events). Please ask a related question.")

    if 'history' in st.session_state and st.session_state.history:
        if st.checkbox("Show Previous Interactions"):
            for idx, interaction in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Interaction {idx + 1}"):
                    st.markdown(f"*User Input:* {interaction['user_input']}")
                    st.markdown(f"*GAILGPT's Response:* {interaction['ai_response']}")

# Main application logic
def main():
    st.title("GailGPT\nIntelligent Enterprise Assistant")

    # Initialize chatbot and models
    conversation, embedding_model = initialize_chatbot()

    # HR, IT, and Company Event keywords
    hr_it_event_keywords = [
        # HR Topics
        "Employee onboarding", "Exit interview", "Compensation policy", "Employee engagement programs",
        "Performance improvement plan", "Payroll management", "Recruitment policy", "Work-from-home policy",
        # IT Support Topics
        "IT helpdesk ticketing system", "Software installation and updates", "Network configuration",
        "Password reset and recovery", "VPN setup and troubleshooting", "Cybersecurity awareness training",
        # Company Events and Meetings
        "Annual general meeting", "Holiday parties", "Team-building activities", "Corporate retreats"
    ]

    # Encode HR, IT, and Event keywords
    category_embeddings = embedding_model.encode(hr_it_event_keywords, convert_to_tensor=True)

    # OTP Authentication
    if not st.session_state.get('authenticated', False):
        authenticate_user()
    else:
        chatbot_interface(conversation, embedding_model, category_embeddings)

if __name__ == "__main__":
    main()
