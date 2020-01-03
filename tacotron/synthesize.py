import argparse
import os
import re
import time
from time import sleep
from datasets import audio
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm


def generate_fast(model, text):
	model.synthesize(text, None, None, None, None)


def run_live(args, checkpoint_path, hparams):
	#Log to Terminal without keeping any records in files
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Generate fast greeting message
	greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
	log(greetings)
	generate_fast(synth, greetings)

	#Interaction loop
	while True:
		try:
			text = input()
			generate_fast(synth, text)

		except KeyboardInterrupt:
			leave = 'Thank you for testing our features. see you soon.'
			log(leave)
			generate_fast(synth, leave)
			sleep(2)
			break

def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	delta_size = hparams.tacotron_synthesis_batch_size if hparams.tacotron_synthesis_batch_size < len(sentences) else len(sentences)
	batch_sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), delta_size)]

	start = time.time()
	for i, batch in enumerate(tqdm(batch_sentences)):
		audio.save_wav(synth.eval(batch), os.path.join(log_dir, 'wavs', 'eval_batch_{:03}.wav'.format(i)), hparams)
	log('\nGenerated batches of size {} in {:.3f} sec'.format(delta_size, time.time() - start))

	return eval_dir


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
	output_dir = 'tacotron_' + args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	if args.mode == 'eval':
		return run_eval(args, checkpoint_path, output_dir, hparams, sentences)
	else:
		run_live(args, checkpoint_path, hparams)
