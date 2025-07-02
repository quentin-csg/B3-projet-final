import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa
from pydub import AudioSegment

VIDEO_LIST = [
    '../video/fake_queen.mp4',
    '../video/fake_greta.mp4',
    '../video/fake_macron.mp4',
    '../video/fake_trump.mp4',
    '../video/fake.mp4',
    '../video/real_trump.mp4',
    '../video/real_obama.mp4',
    '../video/real_macron.mp4',
    '../video/real.mp4'
]

def extract_audio_from_mp4(mp4_path, wav_path):
    audio = AudioSegment.from_file(mp4_path)
    audio.export(wav_path, format="wav")

def analyse_f0_simple(mp4_path):
    wav_path = 'temp_f0.wav'
    extract_audio_from_mp4(mp4_path, wav_path)
    y, sr = librosa.load(wav_path, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=800, sr=sr)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    mask = ~np.isnan(f0)
    f0 = f0[mask]
    times = times[mask]
    if len(f0) == 0:
        return 'INCONNU'
    f0_std = np.std(f0) / np.mean(f0) if np.mean(f0) > 0 else 0
    delta_f0 = np.diff(f0)
    seuil_saut = 50
    sauts_brusques = np.abs(delta_f0) > seuil_saut
    nb_sauts = np.sum(sauts_brusques)
    duree_totale = times[-1] - times[0] if len(times) > 0 else 0
    sauts_par_min = nb_sauts / (duree_totale / 60) if duree_totale > 0 else 0
    delta_t = np.diff(times)
    seuil_trou = 0.5
    trous = delta_t > seuil_trou
    nb_trous = np.sum(trous)
    trous_par_min = nb_trous / (duree_totale / 60) if duree_totale > 0 else 0
    if f0_std == 0 or sauts_par_min > 10 or trous_par_min > 10:
        return 'FAKE'
    return 'REAL'

def read_wav(wav_path):
    samplerate, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return samplerate, data

def analyse_harmony_simple(mp4_path, threshold_db=-80, min_db_limit=-100, ignore_too_low=True):
    wav_path = 'temp_harmony.wav'
    extract_audio_from_mp4(mp4_path, wav_path)
    samplerate, data = read_wav(wav_path)
    nperseg = 1024
    f, t, Sxx = signal.spectrogram(data, samplerate, nperseg=nperseg)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    lowcut = 4000
    highcut = 6000
    freq_mask = (f >= lowcut) & (f <= highcut)
    f_band = f[freq_mask]
    Sxx_band = Sxx_dB[freq_mask, :]
    min_per_time = np.min(Sxx_band, axis=0)
    t_mask = t > 1
    t_after = t[t_mask]
    min_per_time_after = min_per_time[t_mask]
    if ignore_too_low:
        below_threshold = (min_per_time_after <= threshold_db) & (min_per_time_after > min_db_limit)
    else:
        below_threshold = min_per_time_after <= threshold_db
    if np.any(below_threshold):
        return 'FAKE'
    else:
        return 'REAL'

def main():
    results = []
    for video in VIDEO_LIST:
        res_f0 = analyse_f0_simple(video)
        res_harmony = analyse_harmony_simple(video)
        results.append({
            'video': os.path.basename(video),
            'F0': res_f0,
            'Harmony': res_harmony
        })
    print(f"{'Video':40} | {'F0':8} | {'Harmony':8}")
    print('-'*60)
    for r in results:
        print(f"{r['video']:40} | {r['F0']:8} | {r['Harmony']:8}")

if __name__ == '__main__':
    main()