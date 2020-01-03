import librosa
import numpy as np
import pysptk
import pyworld
import soundfile as sf
import tensorflow as tf


def load_wav(path, hparams):
	wav, _ = sf.read(path)
	return wav

def save_wav(wav, path, hparams):
	sf.write(path, wav, hparams.sample_rate)

def trim_silence(wav, hparams):
	return librosa.effects.trim(wav, top_db= hparams.trim_top_db,
		frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def feature_extract(wav, hp):
	fs = hp.sample_rate
	if hp.use_harvest:
		f0, timeaxis = pyworld.harvest(wav, fs, frame_period=hp.frame_period)
	else:
		f0, timeaxis = pyworld.dio(wav, fs, frame_period=hp.frame_period)
		f0 = pyworld.stonemask(wav, f0, timeaxis, fs)

	spectrogram = pyworld.cheaptrick(wav, f0, timeaxis, fs)
	aperiodicity = pyworld.d4c(wav, f0, timeaxis, fs)
	bap = pyworld.code_aperiodicity(aperiodicity, fs)
	hp.num_bap = bap.shape[1]
	alpha = pysptk.util.mcepalpha(fs)
	mgc = pysptk.sp2mc(spectrogram, order=hp.num_mgc - 1, alpha=alpha)
	f0 = f0[:, None]
	lf0 = f0.copy()
	nonzero_indices = np.nonzero(f0)
	lf0[nonzero_indices] = np.log(f0[nonzero_indices])
	if hp.use_harvest:
		# https://github.com/mmorise/World/issues/35#issuecomment-306521887
		vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
	else:
		vuv = (lf0 != 0).astype(np.float32)

	features = np.hstack((mgc, lf0, vuv, bap))
	return features.astype(np.float32)

def synthesize(feature, hparams):
	mgc_idx = 0
	lf0_idx = mgc_idx + hparams.num_mgc
	vuv_idx = lf0_idx + hparams.num_lf0
	bap_idx = vuv_idx + hparams.num_vuv

	mgc = feature[:, mgc_idx : mgc_idx + hparams.num_mgc]
	lf0 = feature[:, lf0_idx : lf0_idx + hparams.num_lf0]
	vuv = feature[:, vuv_idx : vuv_idx + hparams.num_vuv]
	bap = feature[:, bap_idx : bap_idx + hparams.num_bap]

	fs = hparams.sample_rate
	alpha = pysptk.util.mcepalpha(fs)
	fftlen = pyworld.get_cheaptrick_fft_size(fs)

	spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)

	indexes = (vuv < 0.5).flatten()
	bap[indexes] = np.zeros(hparams.num_bap)
	aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)

	f0 = lf0.copy()
	f0[vuv < 0.5] = 0
	f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

	return pyworld.synthesize(f0.flatten().astype(np.float64),
				spectrogram.astype(np.float64),
				aperiodicity.astype(np.float64),
				fs, hparams.frame_period)
