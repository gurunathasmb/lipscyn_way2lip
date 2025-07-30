import pyttsx3
import subprocess

def save_tts(text, output_file="speech.wav"):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    return output_file