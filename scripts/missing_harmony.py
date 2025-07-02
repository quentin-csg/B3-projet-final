import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment
import sys

# Extrait l'audio d'une vidéo MP4 et le sauvegarde en WAV
def extract_audio_from_mp4(mp4_path, wav_path):
    audio = AudioSegment.from_file(mp4_path)
    audio.export(wav_path, format="wav")

# Lit un fichier WAV et retourne l'échantillonnage et les données mono
def read_wav(wav_path):
    samplerate, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return samplerate, data

# Calcule le spectrogramme et vérifie si l'amplitude descend sous un seuil dans une bande de fréquences
def spectrogram_and_check_threshold(mp4_path, output_png, threshold_db=-80, min_db_limit=-100, ignore_too_low=True):
    wav_path = 'temp.wav'
    extract_audio_from_mp4(mp4_path, wav_path)
    samplerate, data = read_wav(wav_path)
    nperseg = 1024
    f, t, Sxx = signal.spectrogram(data, samplerate, nperseg=nperseg)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # Définition de la bande de fréquences d'intérêt
    lowcut = 4000
    highcut = 6000
    freq_mask = (f >= lowcut) & (f <= highcut)
    f_band = f[freq_mask]
    Sxx_band = Sxx_dB[freq_mask, :]
    min_per_time = np.min(Sxx_band, axis=0)

    # On ignore la première seconde
    t_mask = t > 1
    t_after = t[t_mask]
    min_per_time_after = min_per_time[t_mask]

    # Détection des intervalles sous le seuil
    if ignore_too_low:
        below_threshold = (min_per_time_after <= threshold_db) & (min_per_time_after > min_db_limit)
    else:
        below_threshold = min_per_time_after <= threshold_db

    print(f"\nFichier analysé : {mp4_path}")
    if np.any(below_threshold):
        print(f"Oui, l'amplitude est descendue sous {threshold_db} dB dans la bande 4–6 kHz (après 1s).\n❌ FAKE ❌")
        intervals = []
        values = []
        start = None
        current_values = []

        for i, val in enumerate(below_threshold):
            if val:
                if start is None:
                    start = t_after[i]
                current_values.append(min_per_time_after[i])
            elif start is not None:
                intervals.append((start, t_after[i]))
                values.append(current_values)
                start = None
                current_values = []

        if start is not None:
            intervals.append((start, t_after[-1]))
            values.append(current_values)

        print("Plages temporelles (après 1s) où la condition est vérifiée :")
        for (start, end), val_list in zip(intervals, values):
            val_str = ", ".join("{:.2f} dB".format(v) for v in val_list)
            print(f"  - De {start:.2f}s à {end:.2f}s : valeurs min = [{val_str}]")
    else:
        print(f"Non, l'amplitude n'est jamais descendue sous {threshold_db} dB dans la bande 4–6 kHz (après 1s).\n✅ REAL ✅")

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f_band, Sxx_band, shading='auto', cmap='magma')
    plt.ylabel('Fréquence (Hz)')
    plt.xlabel('Temps (s)')
    plt.title('Spectrogramme dans la bande 4–6 kHz')
    plt.colorbar(label='Amplitude (dB)')
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    os.remove(wav_path)

def batch_process(mp4_paths, ignore_too_low=True):
    for mp4_path in mp4_paths:
        base_name = os.path.splitext(os.path.basename(mp4_path))[0]
        output_png = f'spectrogram_{base_name}_4_6kHz.png'
        spectrogram_and_check_threshold(mp4_path, output_png, ignore_too_low=ignore_too_low)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python missing_harmony.py chemin/vers/video.mp4")
        sys.exit(1)
    mp4_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    output_png = f'spectrogram_{base_name}_4_6kHz.png'
    spectrogram_and_check_threshold(mp4_path, output_png, ignore_too_low=True)