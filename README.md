# Détection de Deepfake – Projet Final B3

Ce projet vise à détecter si une vidéo est un deepfake en analysant plusieurs critères audio et vidéo à l'aide de scripts Python.

### Scripts utilisés

- **f0_interpretor.py** : Analyse la fréquence fondamentale de la voix (F0) pour détecter des anomalies caractéristiques des deepfakes (sauts, stabilité anormale, etc).
- **missing_harmony.py** : Analyse le spectre audio pour repérer des pertes d'harmoniques dans la bande 4–6 kHz, souvent présentes dans les deepfakes.
- **clignement_oeil.py** : Détecte les clignements des yeux sur la vidéo pour vérifier la naturalité du comportement facial.
- **sync_levres_voix.py** : Analyse la synchronisation entre les mouvements des lèvres et la piste audio pour repérer des désynchronisations suspectes.

## Installation

```
git clone https://github.com/quentin-csg/B3-projet-final.git
```

## Utilisation

Placez vos vidéos dans le dossier `videos/` puis lancez l'un des scripts comme ceci :
```sh
python .\scripts\nom_du_script.py .\videos\nom_de_la_video.mp4
```

L'analyse sera visiblement dans votre terminal, et pour les 2 scripts d'analyse audio `f0_interpretor.py` et `missing_harmony.py`, un graphique sera également généré au format png.

## Exemple de résultat

```
Fichier analysé : ./video/fake_greta.mp4
Oui, l'amplitude est descendue sous -80 dB dans la bande 4–6 kHz (après 1s).
FAKE
Plages temporelles (après 1s) où la condition est vérifiée :
  - De 46.50s à 46.52s : valeurs min = [-80.86 dB]
  - De 46.56s à 46.60s : valeurs min = [-89.48 dB, -88.75 dB]
```

## Auteurs
- Axel BROQUAIRE
- Quentin CASSAIGNE