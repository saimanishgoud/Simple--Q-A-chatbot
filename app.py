import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
import os
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Gemini Q&A Chatbot", page_icon="🤖")
st.title("🤖 Gemini AI Q&A Chatbot")

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------
st.sidebar.header("⚙️ Model Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Max Output Tokens",
    min_value=100,
    max_value=5000,
    value=1200,
    step=100
)

# ----------------------------
# API KEY
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY as environment variable.")
    st.stop()

# ----------------------------
# LLM INITIALIZATION
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=temperature,
    max_output_tokens=max_tokens,
    
)

# ----------------------------
# SESSION STATE (CHAT HISTORY)
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="""
You are a large language model designed to provide clear, structured, and professional explanations.

For every user question, follow this response structure strictly:

1. Concept Overview  
   - Provide a clear and formal definition.  
   - Explain what the concept is and why it matters.

2. Core Components or Types  
   - Break down the main elements, categories, or architecture (if applicable).

3. Step-by-Step Explanation  
   - Explain how it works logically and sequentially.  
   - If technical, describe the internal process clearly.

4. Practical Applications  
   - Mention real-world or industry use cases.

5. Example  
   - Provide a simple technical example (not a story or analogy).

6. Summary  
   - End with a concise structured recap (3–4 lines).

Guidelines:
- Do not use storytelling or casual tone.
- Do not give one-line answers.
- Use bullet points and sections.
- Keep explanations precise, professional, and technically accurate.
- If unsure, state limitations clearly instead of guessing.
"""
        )
    ]

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.chat_input("Ask me anything about AI, ML, Cloud, etc...")

if user_input and user_input.strip() != "":
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input.strip()))

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)

            st.write(response.content)

    # Add AI response to history
    st.session_state.messages.append(AIMessage(content=response.content))