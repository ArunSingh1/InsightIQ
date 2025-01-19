import streamlit as st
from io import BytesIO
from openai import OpenAI
import os
from dotenv import load_dotenv
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# audio_value = st.audio_input("Record a voice message", label_visibility="hidden")

def audiototextOpenAI(audio_value):

    transcription = None

    if audio_value:
        st.audio(audio_value)
        # Save the audio file to the root folder
        audio_bytes = audio_value.getvalue()
        with open("recorded_audio.wav", "wb") as audio_file:
            audio_file.write(audio_bytes)
        # st.write("Audio file saved as 'recorded_audio.wav'")
        client = OpenAI()
        audio_file= open("recorded_audio.wav", "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        # st.write(transcription.text)

    return transcription