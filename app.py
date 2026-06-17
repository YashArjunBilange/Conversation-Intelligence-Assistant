import os

import streamlit as st

from rag_chatbot import ask_question


st.set_page_config(
    page_title="Conversation Intelligence Assistant",
    page_icon="💬",
    layout="centered",
)

st.title("Conversation Intelligence Assistant")
st.caption("Ask a question and the app will retrieve supporting evidence before answering.")

with st.sidebar:
    st.subheader("Deployment status")
    if os.getenv("GEMINI_API_KEY"):
        st.success("GEMINI_API_KEY is set.")
    else:
        st.warning("Add GEMINI_API_KEY in Streamlit secrets or the environment.")

    st.write("The app rebuilds its vector store from topics.json and message_chunks.json when needed.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about the user")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching conversation history..."):
            answer = ask_question(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})