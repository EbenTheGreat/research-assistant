import streamlit as st
from frontend.utils.api import upload_pdfs

def render_uploader():
    st.sidebar.header("Upload Document (.PDFs)")
    uploaded_files = st.sidebar.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Upload DB") and uploaded_files:
        response = upload_pdfs(uploaded_files)
        if response.status_code == 200:
            st.sidebar.success("Uploaded Successfully!")

        else:
            st.sidebar.error(f"Error: {response.text}")
