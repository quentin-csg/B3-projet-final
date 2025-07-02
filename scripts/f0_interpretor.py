import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from scipy import signal
import librosa
import os
import sys
import matplotlib.pyplot as plt

# Extrait l'audio d'une vidéo MP4 et le sauvegarde en WAV
def extract_audio_from_mp4(mp4_path, wav_path):
    audio = AudioSegment.from_file(mp4_path)
    audio.export(wav_path, format="wav")

# Analyse la fréquence fondamentale (F0) extraite
def analyse_f0(times, f0, sr, mp4_path):
    mask = ~np.isnan(f0)
    times = times[mask]
    f0 = f0[mask]

    if len(f0) == 0:
        print("Aucune F0 détectée. Vérifiez l'audio ou la plage de fréquence.")
        return

    # Calcul des bornes de F0
    f0_min, f0_max = np.min(f0), np.max(f0)
    print(f"Plage de F0 : {f0_min:.1f} Hz à {f0_max:.1f} Hz")

    # Calcul de la variabilité relative de F0
    f0_std = np.std(f0) / np.mean(f0) if np.mean(f0) > 0 else 0
    print(f"Variabilité relative de F0 : {f0_std:.4f}")

    # Détection des sauts brusques de F0
    delta_f0 = np.diff(f0)
    seuil_saut = 50  # en Hz
    sauts_brusques = np.abs(delta_f0) > seuil_saut
    nb_sauts = np.sum(sauts_brusques)
    duree_totale = times[-1] - times[0] if len(times) > 0 else 0
    sauts_par_min = nb_sauts / (duree_totale / 60) if duree_totale > 0 else 0
    print(f"Nombre de sauts brusques de F0 (> {seuil_saut} Hz) : {nb_sauts} (soit {sauts_par_min:.1f} par minute)")

    # Détection des trous anormaux dans la F0
    delta_t = np.diff(times)
    seuil_trou = 0.5  # en secondes
    trous = delta_t > seuil_trou
    nb_trous = np.sum(trous)
    trous_par_min = nb_trous / (duree_totale / 60) if duree_totale > 0 else 0
    print(f"Nombre de trous anormaux dans la F0 (> {seuil_trou}s) : {nb_trous} (soit {trous_par_min:.1f} par minute)")

    print("\nRésumé d'analyse de la F0 :")
    if f0_std < 0.05 and f0_std > 0:
        print("- F0 très stable, vérifier le contexte (peut être naturel pour une voix monotone)")
    if sauts_par_min > 10:
        print("- Nombre élevé de sauts brusques de F0 (risque de deepfake)")
    else:
        print("- Nombre de sauts brusques de F0 dans la norme")
    if trous_par_min > 10:
        print("- Nombre élevé de trous anormaux dans la F0 (risque de deepfake)")
    else:
        print("- Nombre de trous anormaux dans la F0 dans la norme")
    if f0_min < 70 or f0_max > 450:
        print("- Plage de F0 hors norme pour la voix humaine adulte")
    if f0_std == 0:
        print("- F0 parfaitement stable (très rare en parole naturelle, possible en synthèse)")
    if duree_totale == 0 or len(f0) == 0:
        print("- Signal trop court ou F0 non détectée, analyse impossible")

    # Détection si le fichier est supposé réel ou fake
    is_real = "real" in os.path.basename(mp4_path).lower()
    print("\nDiagnostic personnalisé :")
    if is_real:
        print("- Fichier labellisé comme réel.")
        if sauts_par_min > 10 or trous_par_min > 10:
            print("  - ATTENTION : Paramètres F0 atypiques pour une voix réelle.")
    else:
        print("- Fichier labellisé comme fake.")
        if sauts_par_min > 10 or trous_par_min > 10:
            print("  - Paramètres F0 compatibles avec une voix synthétique.")
        else:
            print("  - Paramètres F0 proches d'une voix naturelle.")

def plot_spectrogram_and_f0(mp4_path, analyse=True):
    wav_path = 'temp.wav'
    extract_audio_from_mp4(mp4_path, wav_path)
    samplerate, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    lowcut = 80
    highcut = 800

    # Calcul du spectrogramme
    nperseg = 1024
    f, t, Sxx = signal.spectrogram(data, samplerate, nperseg=nperseg)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    freq_mask = (f >= lowcut) & (f <= highcut)
    f_band = f[freq_mask]
    Sxx_band = Sxx_dB[freq_mask, :]

    y, sr = librosa.load(wav_path, sr=samplerate)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=lowcut, fmax=highcut, sr=sr)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

    if analyse:
        print(f"\nFichier testé : {mp4_path}")
        analyse_f0(times, f0, sr, mp4_path)

    # Visualisation type missing_harmony
    output_png = f"f0_visualisation_{os.path.splitext(os.path.basename(mp4_path))[0]}.png"
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f_band, Sxx_band, shading='auto', cmap='magma', vmin=-100, vmax=50)
    plt.ylabel('Fréquence (Hz)')
    plt.xlabel('Temps (s)')
    plt.title('Spectrogramme vocal + F0 estimée')
    cbar = plt.colorbar(label='Amplitude (dB)')
    mask = ~np.isnan(f0)
    plt.plot(times[mask], f0[mask], 'w-', linewidth=2, label='F0 estimée')
    plt.ylim(lowcut, highcut)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualisation enregistrée dans {output_png}")

    os.remove(wav_path)
    print(f"Fichier testé : {mp4_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python f0_interpretor.py chemin/vers/video.mp4")
        sys.exit(1)
    mp4_path = sys.argv[1]
    filename = os.path.basename(mp4_path)
    plot_spectrogram_and_f0(mp4_path, analyse=True)