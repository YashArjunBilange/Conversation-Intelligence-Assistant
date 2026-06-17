import streamlit as st
from rag_chatbot import ask_question

st.title("Conversation Intelligence Assistant")

question = st.text_input(
    "Ask a question about the user"
)

if st.button("Ask"):
    answer = ask_question(question)
    st.write(answer)