import torch
import librosa
# import soundfile
import numpy as np
import torch.nn as nn
import torchaudio.transforms as T
from amd.amd_module import PreEmphasis


def add_noise(x, snr, method='vectorized', axis=0):
    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)
    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))
    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)
    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    Psdb = 10 * np.log10(Ps)        # Signal power, in dB
    Pn = Psdb - snr         # Noise level necessary
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)      # Noise vector (or matrix)
    return x + n


def load_spectrogram(wav):
    mag = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=512, n_mels=80,
                                         win_length=400, hop_length=160)
    mag = librosa.power_to_db(mag, ref=1.0, amin=1e-10, top_db=None)
    return mag


def audio_to_wav(filename, sr=16000, noise=False):
    wav, fs = librosa.load(filename, sr=sr)
    # wav = wav.astype(np.double)
    extended_wav = np.append(wav, wav)
    if noise:
        extended_wav = add_noise(extended_wav, fs)
    return extended_wav, fs


def loadWAV(filename, noise=False):
    y, sr = audio_to_wav(filename=filename, noise=noise)

    tiny = nn.Sequential(
        PreEmphasis(),
        T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                         f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80)
    )
    y = torch.from_numpy(y).unsqueeze(0).float()
    y = tiny(y).squeeze(0) + 1e-6
    y = y.log()
    y = y - torch.mean(y, dim=-1, keepdim=True)

    # y = load_spectrogram(y)
    # y = torch.from_numpy(y).float()
    return y


if __name__ == '__main__':
    audio_filename = "..\\data\\test\\G2231\\T0055G2231S0076.wav"
    a = loadWAV(audio_filename, noise=True)
    print(a.shape, a.dtype)
