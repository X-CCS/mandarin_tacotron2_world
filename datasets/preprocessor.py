import glob, os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datasets import audio


def build_from_path(hparams, input_dirs, feat_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		print("input_dirs:",input_dirs)
		print("input_dir:",input_dir)
		
		# trn_files = glob.glob(os.path.join(input_dir, 'biaobei_48000', '*.trn'))
		trn_files = glob.glob(os.path.join(input_dir, 'data', '*.trn'))
		for trn in trn_files:
			print("trn:",trn)
			with open(trn) as f:
				basename = trn[:-4]
				print("basename:",basename)
				# wav_file = basename + '.wav'
				wav_file = basename
				wav_path = wav_file
				print("wav_path:",wav_path)
				basename = basename.split('/')[-1]
				text = f.readline().strip()
				futures.append(executor.submit(partial(_process_utterance, feat_dir, basename, wav_path, text, hparams)))
				index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(feat_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, hparams)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
		return None

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	# feature extraction
	feature = audio.feature_extract(wav, hparams)
	n_frames = len(feature)
	if n_frames > hparams.max_frame_num or len(text) > hparams.max_text_length:
		return None
	
	feat_file = '{}.npy'.format(index)
	np.save(os.path.join(feat_dir, feat_file), feature, allow_pickle=False)

	# Return a tuple describing this training example
	return (feat_file, n_frames, text)
