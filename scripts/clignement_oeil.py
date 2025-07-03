import cv2
import dlib
import numpy as np
import sys

SEUIL_OUVERTURE_OEIL = 0.25
NB_IMAGES_CONSECUTIVES = 2
TAUX_CLIGNEMENTS_NORMAL = (8, 28)  # clignements/min

# Initialisation du détecteur et du predictisseur de points de repère
detecteur_visage = dlib.get_frontal_face_detector()
predicteur_points = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(points_oeil):
    # Calcul de la EAR (Eye Aspect Ratio)
    A = np.linalg.norm(points_oeil[1] - points_oeil[5])
    B = np.linalg.norm(points_oeil[2] - points_oeil[4])
    C = np.linalg.norm(points_oeil[0] - points_oeil[3])
    return (A + B) / (2 * C)

# Ouverture du flux vidéo
capture = cv2.VideoCapture(sys.argv[1])
images_par_seconde = capture.get(cv2.CAP_PROP_FPS)
compteur_images = 0
clignements_total = 0
compte_consecutif = 0

while True:
    ret, image = capture.read()
    if not ret:
        break

    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visages = detecteur_visage(gris)

    if visages:
        forme = predicteur_points(gris, visages[0])
        points_repere = forme.parts()
        coordonnees = []
        for pt in points_repere:
            coordonnees.append((pt.x, pt.y))
        tableau_coords = np.array(coordonnees)

        oeil_droit = tableau_coords[36:42]  # indice oeil droit
        oeil_gauche = tableau_coords[42:48] # indice oeil gauche

        ear_left = eye_aspect_ratio(oeil_gauche)
        ear_right = eye_aspect_ratio(oeil_droit)
        ear_moyen = (ear_left + ear_right) / 2

        if ear_moyen < SEUIL_OUVERTURE_OEIL:
            compte_consecutif += 1
        elif compte_consecutif >= NB_IMAGES_CONSECUTIVES:
            clignements_total += 1
            compte_consecutif = 0
        else:
            compte_consecutif = 0

    compteur_images += 1

capture.release()

# Calcul du taux de clignements par minute
minutes_enregistrement = compteur_images / images_par_seconde / 60
if minutes_enregistrement != 0:
    taux_clignements = clignements_total / minutes_enregistrement
else:
    taux_clignements = 0

# Détermination du statut
if TAUX_CLIGNEMENTS_NORMAL[0] <= taux_clignements <= TAUX_CLIGNEMENTS_NORMAL[1]:
    etat = "NORMAL"
else:
    etat = "SUSPECT"
print(f"Clignements détectés: {clignements_total}")
print(f"Taux: {taux_clignements:.1f} clign./min → {etat}")
