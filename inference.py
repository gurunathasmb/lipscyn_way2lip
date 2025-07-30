from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
import platform

parser = argparse.ArgumentParser(description='Inference code for Wav2Lip with TorchScript model')

parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to TorchScript .pt model file')
parser.add_argument('--face', type=str, required=True, help='Path to video/image with face')
parser.add_argument('--audio', type=str, required=True, help='Path to audio or video file for lip sync')
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4', help='Output video path')

parser.add_argument('--static', type=bool, default=False, help='Use only the first frame')
parser.add_argument('--fps', type=float, default=25., help='FPS for static images')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, default=16)
parser.add_argument('--wav2lip_batch_size', type=int, default=128)
parser.add_argument('--resize_factor', type=int, default=1)
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
parser.add_argument('--rotate', default=False, action='store_true')
parser.add_argument('--nosmooth', default=False, action='store_true')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
	args.static = True

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		window = boxes[max(0, i - T//2):min(len(boxes), i + T//2)]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
	batch_size = args.face_det_batch_size
	while True:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('OOM in face detection. Use --resize_factor.')
			batch_size //= 2
			print('Reduced batch size to:', batch_size)
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image)
			raise ValueError('Face not detected. Check temp/faulty_frame.jpg')
		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth:
		boxes = get_smoothened_boxes(boxes, T=5)
	return [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
	        for image, (x1, y1, x2, y2) in zip(images, boxes)]

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	if args.box[0] == -1:
		face_det_results = face_detect([frames[0]]) if args.static else face_detect(frames)
	else:
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, mel in enumerate(mels):
		idx = 0 if args.static else i % len(frames)
		face, coords = face_det_results[idx].copy()
		face = cv2.resize(face, (args.img_size, args.img_size))

		img_batch.append(face)
		mel_batch.append(mel)
		frame_batch.append(frames[idx].copy())
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			yield prepare_batch(img_batch, mel_batch), frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		yield prepare_batch(img_batch, mel_batch), frame_batch, coords_batch

def prepare_batch(img_batch, mel_batch):
	img_batch = np.asarray(img_batch)
	mel_batch = np.asarray(mel_batch)
	img_masked = img_batch.copy()
	img_masked[:, args.img_size//2:] = 0
	img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
	mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
	return img_batch, mel_batch

def load_model(path):
	print("Loading TorchScript model:", path)
	model = torch.jit.load(path, map_location=device)
	return model.eval()

def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face must be a valid file path')

	if args.static:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps
	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		full_frames = []
		while True:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
			if args.rotate:
				frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]
			full_frames.append(frame[y1:y2, x1:x2])

	print("Total frames:", len(full_frames))

	if not args.audio.endswith('.wav'):
		print("Extracting audio...")
		command = f'ffmpeg -y -i "{args.audio}" -strict -2 temp/temp.wav'
		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError("Mel contains NaN values.")

	mel_chunks = []
	mel_idx_multiplier = 80. / fps
	i = 0
	while True:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > mel.shape[1]:
			mel_chunks.append(mel[:, -mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
		i += 1

	full_frames = full_frames[:len(mel_chunks)]

	model = load_model(args.checkpoint_path)
	print("Model loaded.")

	gen = datagen(full_frames.copy(), mel_chunks)
	frame_h, frame_w = full_frames[0].shape[:-1]
	out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

	for i, ((img_batch, mel_batch), frames, coords) in enumerate(
		tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / args.wav2lip_batch_size)))):
		
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()
	command = f'ffmpeg -y -i "{args.audio}" -i "temp/result.avi" -strict -2 -q:v 1 "{args.outfile}"'
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
