import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatui import render_chat
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title(" 🤖 Research Assistant Chatbot")


render_uploader()
render_chat()
render_history_download()











