import os
import wave
from datetime import datetime
import numpy as np
import sounddevice as sd
import tensorflow as tf
from datasets import audio
from infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence


class Synthesizer:
	def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
		log('Constructing model: %s' % model_name)
		inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
		input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
		with tf.variable_scope('model') as scope:
			self.model = create_model(model_name, hparams)
			self.model.initialize(inputs, input_lengths)
			self.final_outputs = self.model.final_outputs
			self.alignments = self.model.alignments
			self.stop_token_outputs = self.model.stop_token_outputs

		self.gta = gta
		self._hparams = hparams
		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPU as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames, out_dir, log_dir):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		features, alignments, stop_tokens = self.session.run([self.final_outputs, self.alignments, self.stop_token_outputs], feed_dict=feed_dict)

		#Get feature output lengths for the entire batch from stop_tokens outputs
		output_lengths = self._get_output_lengths(stop_tokens)
		features = [feature[:output_length, :] for feature, output_length in zip(features, output_lengths)]
		assert len(features) == len(texts)

		for i, feature in enumerate(features):
			# Write the predicted features to disk
			# Note: outputs files and target ones have same names, just different folders
			np.save(os.path.join(out_dir, 'feature-{:03d}.npy'.format(i+1)), feature, allow_pickle=False)

			if log_dir is not None:
				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{:03d}.png'.format(i+1)),
					info='{}'.format(texts[i]), split_title=True)

				#save wav
				wav = audio.synthesize(feature, hparams)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{:03d}.wav'.format(i+1)), hparams)


	def eval(self, batch):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in batch]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}
		features, stop_tokens = self.session.run([self.model.final_outputs, self.stop_token_outputs], feed_dict=feed_dict)

		#Get feature output lengths for the entire batch from stop_tokens outputs
		output_lengths = self._get_output_lengths(stop_tokens)
		features = [feature[:output_length, :] for feature, output_length in zip(features, output_lengths)]
		assert len(features) == len(batch)

		wavs = []
		for i, feature in enumerate(features):
			np.save('tacotron_output/{}.npy'.format(i+1), feature)
			wavs.append(audio.synthesize(feature, hparams))
		return np.concatenate(wavs)


	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		return np.stack([self._pad_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _get_output_lengths(self, stop_tokens):
		#Determine each mel length by the stop token outputs. (len = first occurence of 1 in stop_tokens row wise)
		output_lengths = [row.index(1) + 1 if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
		return output_lengths
