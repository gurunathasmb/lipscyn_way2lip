import subprocess
import os

def run_wav2lip(face_video, audio_path, output_path):
    # Ensure the paths are correct and files exist
    if not os.path.exists(face_video):
        raise FileNotFoundError(f"Face video not found: {face_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    command = [
        "python", "inference.py",
        "--checkpoint_path", "checkpoints/Wav2Lip-SD-GAN.pt",
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path
    ]

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)  # check=True will raise error if it fails
