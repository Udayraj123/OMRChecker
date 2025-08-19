import streamlit as st
from src.docgen import generate_comments
from src.file_utils import read_file, write_file, save_changes
from src.change_tracker import show_changes
import os

st.title("Code Documentation Generator")

# Ask the user for their API key
api_key = st.text_input("Enter your Google API Key:", type="password")

if api_key:
    # Upload code files
    uploaded_files = st.file_uploader("Upload code files", accept_multiple_files=True, type=["py"])

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files. Processing...")

        # Process each file
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")

            # Save the uploaded file temporarily
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Read the original code
            original_code = read_file(file_path)

            # Generate updated code with comments/docs
            updated_code = generate_comments(original_code, api_key)

            # Show changes to the user
            changes = show_changes(original_code, updated_code)
            st.text_area(f"Changes for {uploaded_file.name}", changes, height=300)

            # Ask for user confirmation
            if st.button(f"Apply changes to {uploaded_file.name}"):
                # Save the updated code
                write_file(file_path, updated_code)

                # Save the changes for future reference
                save_changes(uploaded_file.name, changes)

                st.success(f"Changes saved for {uploaded_file.name}!")
else:
    st.warning("Please enter your API key to proceed.")