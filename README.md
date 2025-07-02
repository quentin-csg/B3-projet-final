## Projet finale B3

Notre projet B3 va porter sur la création d'un script qui analysera une vidéo pour savoir si c'est un deepfake ou non en se basant sur plusieurs critères.

Le script principal "jsp.py" va exécuter 4 autres scripts pour avoir un retour globale et déterminer sur la vidéo est considéré comme un deepfake ou non.

- f0_interpretor.py : jsp ce qu'il faut
- missing_harmony.py : jsp nn plus
- clignement_oeil.py : va détecter les clignements des yeux d'une personne sur la vidéo
- sync_levres_voix : analyse les mouvement des lèvres avec la piste audio et calcul la différence entre les deux

Le script va donc retourner un tableau finale avec ces quatres paramètres pris en compte pour déterminer si la vidéo analysé est un deepfake.