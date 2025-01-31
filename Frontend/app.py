import streamlit as st
import requests

# Replace with your backend URL
backend_url = "http://localhost:8000/generate_video"

# Streamlit App
st.title("Text-to-Video Generation Chatbot")
st.write("Enter a text prompt, and the system will generate a video for you using Runway ML!")

# Input field for user prompt
user_input = st.text_input("Enter your prompt:")

# Submit button
if st.button("Submit"):
    if user_input.strip():
        try:
            # Send user input to the backend
            with st.spinner("Processing..."):
                response = requests.post(
                    backend_url,
                    json={"user_input": user_input}
                )
                if response.status_code == 200:
                    data = response.json()
                    video_url = data.get("video_url")

                    if video_url:
                        st.success("Processing Complete!")
                        st.video(video_url)
                    else:
                        st.error("Failed to retrieve the video URL.")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
    else:
        st.warning("Please enter a valid input.")
