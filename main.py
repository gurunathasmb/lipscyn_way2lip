from text_to_speech import save_tts
from image_to_video import image_to_video
from run_wav2lip import run_wav2lip

# === Inputs ===
image_path = "lip.png"   # Place your image here
text = "hello world Text to be converted to speech"

# === File names ===
audio_file = "speech1.wav"
video_file = "face_video1.mp4"
output_file = "final_output1.mp4"

# === Pipeline ===
print("🔊 Generating speech...")
save_tts(text, output_file=audio_file)

print("🎞 Creating video from image...")
image_to_video(image_path, output_video=video_file)

print("🧠 Running Wav2Lip for lip sync...")
run_wav2lip(face_video=video_file, audio_path=audio_file, output_path=output_file)

print(f"✅ Done! Final video saved as: {output_file}")