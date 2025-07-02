import cv2
import dlib
import numpy as np
import librosa
import subprocess
import os
import sys
from scipy.signal import correlate

PREDICTEUR_POINTS = "../shape_predictor_68_face_landmarks.dat"
INDICE_BOUCHE = slice(48, 68)  # indices de la bouche  
DECALAGE_MAX_SEC = 1.0      # recherche de décalage dans ±1s
SEUIL_DECALAGE_SEC = 0.2    # seuil d’alerte en secondes


def extraire_mar(chemin_video):
    # Ouverture de la vidéo
    capture = cv2.VideoCapture(chemin_video)
    images_par_seconde = capture.get(cv2.CAP_PROP_FPS) or 25

    # Initialisation du détecteur de visages et du prédicteur de points
    detecteur_visage = dlib.get_frontal_face_detector()
    predicteur_points = dlib.shape_predictor(PREDICTEUR_POINTS)

    valeurs_mar = []
    instants = []
    derniere_mar = 0.0
    index_image = 0

    while True:
        ret, image = capture.read()
        if not ret:
            break

        temps = index_image / images_par_seconde
        instants.append(temps)

        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        visages = detecteur_visage(gris)

        if visages:
            forme = predicteur_points(gris, visages[0])
            points_repere = forme.parts()
            coordonnees = []
            for pt in points_repere[INDICE_BOUCHE]:
                coordonnees.append((pt.x, pt.y))
            points = np.array(coordonnees)

            # Distances verticale et horizontale de la bouche
            vert1 = np.linalg.norm(points[3] - points[11])
            vert2 = np.linalg.norm(points[5] - points[9])
            horiz = np.linalg.norm(points[0] - points[6])

            if horiz > 0:
                mar = (vert1 + vert2) / (2 * horiz)
                derniere_mar = mar
            else:
                mar = derniere_mar
        else:
            # Pas de visage détecté, on réutilise la dernière valeur
            mar = derniere_mar

        valeurs_mar.append(mar)
        index_image += 1

    capture.release()
    return np.array(instants), np.array(valeurs_mar)


def extraire_rms_audio(chemin_video, fichier_temporaire="tmp.wav", hop_len=512):
    # Extraction de l'audio en WAV via ffmpeg
    if os.path.exists(fichier_temporaire):
        os.remove(fichier_temporaire)
    subprocess.run([
        "ffmpeg", "-y", "-i", chemin_video,
        "-vn",               # ignorer la vidéo
        "-acodec", "pcm_s16le",
        "-ar", "44100",    # 44.1 kHz
        "-ac", "1",        # mono
        fichier_temporaire
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    y, sr = librosa.load(fichier_temporaire, sr=None)
    rms = librosa.feature.rms(y=y, frame_length=hop_len*2, hop_length=hop_len)[0]
    instants_audio = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len, n_fft=hop_len*2)
    return np.array(instants_audio), rms


def calculer_decalage(video_t, video_v, audio_t, audio_v):
    # Intervalle commun
    debut = max(video_t[0], audio_t[0])
    fin = min(video_t[-1], audio_t[-1])

    # Grille de temps uniforme
    pas = np.median(np.diff(video_t))
    temps_commun = np.arange(debut, fin, pas)

    # Interpolation sur la grille
    v_vid = np.interp(temps_commun, video_t, video_v)
    v_aud = np.interp(temps_commun, audio_t, audio_v)

    # Centrage des séries
    v_vid -= v_vid.mean()
    v_aud -= v_aud.mean()

    # Corrélation croisée
    corr = correlate(v_vid, v_aud, mode='full')
    milieu = len(corr) // 2
    decalage_max = int(DECALAGE_MAX_SEC / pas)
    fenetre = corr[milieu-decalage_max : milieu+decalage_max+1]
    indice_max = np.argmax(fenetre) - decalage_max
    return indice_max * pas


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python lip_sync_compact.py video.mp4")
        sys.exit(1)

    chemin_video = sys.argv[1]
    print("Extraction du Mouth Aspect Ratio (MAR)")
    inst_vid, val_mar = extraire_mar(chemin_video)

    print("Extraction du RMS audio")
    inst_aud, val_rms = extraire_rms_audio(chemin_video)

    print("Calcul du décalage audio-vidéo")
    decalage = calculer_decalage(inst_vid, val_mar, inst_aud, val_rms)
    print(f"Décalage estimé : {decalage:.3f} secondes")
    
    if abs(decalage) > SEUIL_DECALAGE_SEC:
        etat = ("POSSIBLE désynchronisation")
    else:
        etat = ("Lip-sync OK")
    print(etat)
