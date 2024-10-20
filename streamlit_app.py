import streamlit as st
import requests

st.title("Resume Query Assistant")

# Text input for user question
user_input = st.text_input("Ask your question about resumes:")

# Submit button
if st.button("Submit"):
    if user_input:  # Check if the input is not empty
        response = requests.post("http://localhost:8000/query/", json={"question": user_input})
        
        # Check the response status
        if response.status_code == 200:
            st.write("Bot:", response.json().get("response", "No response generated."))
        else:
            try:
                error_message = response.json()  # Try to get error message
                st.write("Error:", error_message)
            except ValueError:
                st.write("Error:", response.text)  # Print raw response text
    else:
        st.warning("Please enter a question before submitting.")



