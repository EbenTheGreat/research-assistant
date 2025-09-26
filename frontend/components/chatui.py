import streamlit as st
from frontend.utils.api import ask_questions_stream
import json

def render_chat():
    st.title("🤖 Research Assistant Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    if question := st.chat_input("Ask me anything about your PDFs..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer = ""
            sources = None

            for chunk in ask_questions_stream(question):
                if chunk.startswith("[SOURCES]"):
                    sources = json.loads(chunk[len("[SOURCES]"):])
                elif chunk.startswith("[ERROR]"):
                    answer += f"\n\n Error: {chunk[len('[ERROR]'):]}"
                    placeholder.markdown(answer)
                    break
                else:
                    # Model text
                    answer += chunk
                    placeholder.markdown(answer)

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )

            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.markdown(f"**Source:** {s['source']} (page {s['page']})")
                        st.caption(s["content"][:300] + "...")
