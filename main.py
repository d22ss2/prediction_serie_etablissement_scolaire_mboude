"""
=====================================================================
TP INF232 - Statistiques et Analyse de Données
Thème D : Établissement scolaire secondaire (Lycée de MBOUDA)

Ce fichier contient TOUT le backend de l'application.
Il est volontairement écrit de façon simple, avec des fonctions
courtes et beaucoup de commentaires, pour être facile à expliquer
lors d'une soutenance.

Aucune base de données n'est utilisée : les données restent en
mémoire, dans une simple variable Python (une liste de dictionnaires
et un DataFrame pandas).
=====================================================================
"""

import unicodedata          # pour supprimer les accents d'un texte
import zlib                 # pour calculer une seed numérique stable
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =====================================================================
# 1. CREATION DE L'APPLICATION FASTAPI
# =====================================================================

app = FastAPI(title="TP INF232 - Analyse des élèves du Lycée de MBOUDA")

# On autorise le frontend (fichier index.html ouvert dans le navigateur)
# à communiquer avec ce backend, même s'ils ne sont pas sur le même port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 2. STOCKAGE EN MEMOIRE
# =====================================================================
# On garde ici le DataFrame généré. Comme il n'y a pas de base de
# données, on utilise simplement une variable globale.

donnees_eleves: pd.DataFrame = pd.DataFrame()
seed_actuelle: int = 0


# =====================================================================
# 3. FONCTIONS UTILITAIRES
# =====================================================================

def nettoyer_nom(nom: str) -> str:
    """
    Nettoie le nom saisi par l'utilisateur :
    1. supprime les accents
    2. supprime les espaces
    3. transforme en MAJUSCULES
    """
    # Étape 1 : enlever les accents (é -> e, à -> a, etc.)
    nom_sans_accent = unicodedata.normalize("NFKD", nom)
    nom_sans_accent = "".join(
        caractere for caractere in nom_sans_accent if not unicodedata.combining(caractere)
    )

    # Étape 2 : enlever tous les espaces
    nom_sans_espace = nom_sans_accent.replace(" ", "")

    # Étape 3 : mettre en majuscules
    nom_final = nom_sans_espace.upper()

    return nom_final


def calculer_seed(nom: str) -> int:
    """
    Transforme un nom (déjà nettoyé) en un nombre entier (la seed).
    On utilise crc32, qui donne toujours le même nombre pour le même
    texte. Ainsi, le même nom produira toujours les mêmes données.
    """
    nom_nettoye = nettoyer_nom(nom)
    # encode() transforme le texte en bytes, nécessaire pour crc32
    seed = zlib.crc32(nom_nettoye.encode("utf-8"))
    return seed


def generer_donnees_eleves(nom: str) -> pd.DataFrame:
    """
    Génère un jeu de données de 200 élèves de façon déterministe.
    Plus un élève travaille, plus sa note a tendance à être élevée
    (avec un peu de bruit aléatoire pour que ce soit réaliste).
    """
    global seed_actuelle

    seed = calculer_seed(nom)
    seed_actuelle = seed

    # On fixe la seed de numpy : à partir d'ici, tous les tirages
    # aléatoires seront toujours les mêmes pour cette seed.
    np.random.seed(seed)

    nombre_eleves = 200

    # 1) Heures de travail personnel par semaine : entre 0 et 20 heures
    heures_travail = np.random.uniform(0, 20, nombre_eleves)

    # 2) Note de mathématiques : dépend des heures de travail + bruit
    #    On part d'une base de 5/20, on ajoute un bonus lié au travail,
    #    puis on ajoute un bruit aléatoire (variations individuelles).
    bruit = np.random.normal(0, 2.5, nombre_eleves)
    note_math = 5 + (heures_travail * 0.65) + bruit

    # On garde les notes entre 0 et 20 (une note ne peut pas dépasser 20)
    note_math = np.clip(note_math, 0, 20)
    note_math = np.round(note_math, 2)
    heures_travail = np.round(heures_travail, 1)

    # 3) Orientation : cohérente avec les performances, mais pas
    #    totalement déterministe (comme dans la vraie vie).
    #    On calcule une "probabilité d'aller en filière scientifique"
    #    qui augmente avec la note, puis on tire au sort selon cette
    #    probabilité (fonction logistique).
    probabilite_scientifique = 1 / (1 + np.exp(-(note_math - 11) * 0.5))
    tirage_aleatoire = np.random.uniform(0, 1, nombre_eleves)

    orientation = np.where(
        tirage_aleatoire < probabilite_scientifique,
        "Filière Scientifique",
        "Filière Littéraire",
    )

    # On construit le DataFrame final
    df = pd.DataFrame({
        "id": range(1, nombre_eleves + 1),
        "note_math": note_math,
        "heures_travail": heures_travail,
        "orientation": orientation,
    })

    return df


def vider_valeurs_numpy(valeur):
    """
    Petite fonction utilitaire : convertit les types numpy (np.float64,
    np.int64...) en types Python classiques (float, int), car FastAPI
    ne sait pas toujours convertir les types numpy en JSON.
    """
    if isinstance(valeur, (np.integer,)):
        return int(valeur)
    if isinstance(valeur, (np.floating,)):
        return float(valeur)
    if isinstance(valeur, (np.ndarray,)):
        return valeur.tolist()
    return valeur


# =====================================================================
# 4. MODELES DE DONNEES (pour valider les entrées des routes POST)
# =====================================================================

class DemandeGeneration(BaseModel):
    nom: str


class DemandePrediction(BaseModel):
    note: float
    heures: float


# =====================================================================
# 5. ROUTE : GENERATION DES DONNEES
# =====================================================================

@app.post("/api/generate")
def generer_donnees(demande: DemandeGeneration):
    """
    Reçoit le nom du chef de groupe, génère 200 élèves de façon
    déterministe, et retourne la seed + le jeu de données complet.
    """
    global donnees_eleves

    df = generer_donnees_eleves(demande.nom)
    donnees_eleves = df  # on garde les données en mémoire pour les autres routes

    return {
        "seed": seed_actuelle,
        "nombre_eleves": len(df),
        "donnees": df.to_dict(orient="records"),
    }


# =====================================================================
# 6. ROUTE : ANALYSE DESCRIPTIVE (statistique à une variable)
# =====================================================================

@app.get("/api/statistics/univariate")
def statistiques_univariees():
    """
    Calcule les statistiques descriptives de la note de mathématiques :
    moyenne, médiane, variance, écart-type, min, max, Q1, Q3.
    Retourne aussi les données nécessaires pour tracer un histogramme
    et un boxplot côté frontend.
    """
    if donnees_eleves.empty:
        return {"erreur": "Aucune donnée générée. Veuillez d'abord générer les données."}

    notes = donnees_eleves["note_math"]

    moyenne = notes.mean()
    mediane = notes.median()
    variance = notes.var()
    ecart_type = notes.std()
    minimum = notes.min()
    maximum = notes.max()
    q1 = notes.quantile(0.25)
    q3 = notes.quantile(0.75)

    # Histogramme : on découpe les notes en 10 tranches (bins)
    effectifs, bornes = np.histogram(notes, bins=10, range=(0, 20))

    # Interprétation automatique en langage simple, pour le conseil pédagogique
    if moyenne >= 14:
        niveau = "Le niveau général de la classe est très bon."
    elif moyenne >= 10:
        niveau = "Le niveau général de la classe est moyen, avec une marge de progression possible."
    else:
        niveau = "Le niveau général de la classe est faible et nécessite un accompagnement renforcé."

    if ecart_type >= 4:
        dispersion = "Les résultats sont très dispersés : il existe de fortes disparités entre élèves."
    else:
        dispersion = "Les résultats sont relativement homogènes entre les élèves."

    interpretation = f"{niveau} {dispersion}"

    return {
        "moyenne": vider_valeurs_numpy(round(moyenne, 2)),
        "mediane": vider_valeurs_numpy(round(mediane, 2)),
        "variance": vider_valeurs_numpy(round(variance, 2)),
        "ecart_type": vider_valeurs_numpy(round(ecart_type, 2)),
        "minimum": vider_valeurs_numpy(round(minimum, 2)),
        "maximum": vider_valeurs_numpy(round(maximum, 2)),
        "q1": vider_valeurs_numpy(round(q1, 2)),
        "q3": vider_valeurs_numpy(round(q3, 2)),
        "histogramme": {
            "effectifs": vider_valeurs_numpy(effectifs),
            "bornes": vider_valeurs_numpy(np.round(bornes, 1)),
        },
        "boxplot": {
            "min": vider_valeurs_numpy(round(minimum, 2)),
            "q1": vider_valeurs_numpy(round(q1, 2)),
            "mediane": vider_valeurs_numpy(round(mediane, 2)),
            "q3": vider_valeurs_numpy(round(q3, 2)),
            "max": vider_valeurs_numpy(round(maximum, 2)),
        },
        "interpretation": interpretation,
    }


# =====================================================================
# 7. ROUTE : ANALYSE BIVARIEE (relation entre deux variables)
# =====================================================================

@app.get("/api/statistics/bivariate")
def statistiques_bivariees():
    """
    Étudie la relation entre les heures de travail (X) et la note de
    mathématiques (Y) : covariance, corrélation, régression linéaire,
    coefficient R².
    """
    if donnees_eleves.empty:
        return {"erreur": "Aucune donnée générée. Veuillez d'abord générer les données."}

    x = donnees_eleves["heures_travail"]
    y = donnees_eleves["note_math"]

    covariance = x.cov(y)
    correlation = x.corr(y)

    # Régression linéaire simple : note = a * heures + b
    x_matrice = x.values.reshape(-1, 1)  # scikit-learn attend un tableau 2D
    modele = LinearRegression()
    modele.fit(x_matrice, y.values)

    pente = modele.coef_[0]           # coefficient "a"
    ordonnee_origine = modele.intercept_  # coefficient "b"
    r_carre = modele.score(x_matrice, y.values)  # coefficient R²

    equation = f"note_estimee = {pente:.3f} * heures_travail + {ordonnee_origine:.3f}"

    # On envoie les points au frontend pour tracer le nuage de points
    points = [
        {"x": vider_valeurs_numpy(h), "y": vider_valeurs_numpy(n)}
        for h, n in zip(x, y)
    ]

    return {
        "covariance": vider_valeurs_numpy(round(covariance, 3)),
        "correlation": vider_valeurs_numpy(round(correlation, 3)),
        "pente": vider_valeurs_numpy(round(pente, 3)),
        "ordonnee_origine": vider_valeurs_numpy(round(ordonnee_origine, 3)),
        "r_carre": vider_valeurs_numpy(round(r_carre, 3)),
        "equation": equation,
        "points": points,
    }


# =====================================================================
# 8. ROUTE : CLASSIFICATION NON SUPERVISEE (KMeans)
# =====================================================================

@app.post("/api/cluster")
def classification_non_supervisee():
    """
    Applique un KMeans sur (note_math, heures_travail).
    Teste plusieurs valeurs de k (nombre de groupes) et garde celle qui
    donne le meilleur score de silhouette.
    """
    if donnees_eleves.empty:
        return {"erreur": "Aucune donnée générée. Veuillez d'abord générer les données."}

    x = donnees_eleves[["note_math", "heures_travail"]].values

    meilleur_k = 2
    meilleur_score = -1
    meilleur_modele = None

    # On teste k = 2, 3 et 4 groupes, et on garde le meilleur score
    for k in [2, 3, 4]:
        modele_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquettes = modele_kmeans.fit_predict(x)
        score = silhouette_score(x, etiquettes)

        if score > meilleur_score:
            meilleur_score = score
            meilleur_k = k
            meilleur_modele = modele_kmeans

    etiquettes_finales = meilleur_modele.predict(x)
    centres = meilleur_modele.cluster_centers_

    # On décrit chaque groupe en fonction de la note moyenne du groupe
    descriptions = {}
    for numero_groupe in range(meilleur_k):
        notes_du_groupe = donnees_eleves["note_math"][etiquettes_finales == numero_groupe]
        moyenne_groupe = notes_du_groupe.mean()

        if moyenne_groupe >= 14:
            description = "Très bons élèves"
        elif moyenne_groupe >= 10:
            description = "Élèves moyens"
        else:
            description = "Élèves en difficulté"

        descriptions[str(numero_groupe)] = description

    # Points avec leur groupe, pour le nuage de points coloré
    points = []
    for i in range(len(donnees_eleves)):
        points.append({
            "x": vider_valeurs_numpy(donnees_eleves["heures_travail"].iloc[i]),
            "y": vider_valeurs_numpy(donnees_eleves["note_math"].iloc[i]),
            "groupe": vider_valeurs_numpy(etiquettes_finales[i]),
        })

    centres_liste = [
        {"note_math": vider_valeurs_numpy(c[0]), "heures_travail": vider_valeurs_numpy(c[1])}
        for c in centres
    ]

    return {
        "nombre_groupes": meilleur_k,
        "score_silhouette": vider_valeurs_numpy(round(meilleur_score, 3)),
        "centres": centres_liste,
        "descriptions": descriptions,
        "points": points,
    }


# =====================================================================
# 9. ROUTE : CLASSIFICATION SUPERVISEE (DecisionTreeClassifier)
# =====================================================================

@app.post("/api/classify/train-and-predict")
def classification_supervisee(demande: DemandePrediction):
    """
    Entraîne un arbre de décision (DecisionTreeClassifier) pour prédire
    l'orientation (Scientifique / Littéraire) à partir de la note et
    des heures de travail. Découpe les données en 80% entraînement et
    20% test, puis calcule les indicateurs de performance.
    Enfin, prédit l'orientation pour le nouvel élève fourni.
    """
    if donnees_eleves.empty:
        return {"erreur": "Aucune donnée générée. Veuillez d'abord générer les données."}

    x = donnees_eleves[["note_math", "heures_travail"]].values
    y = donnees_eleves["orientation"].values

    # Découpage 80% entraînement / 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    modele = DecisionTreeClassifier(random_state=42, max_depth=4)
    modele.fit(x_train, y_train)

    # Évaluation du modèle sur les données de test
    y_predit = modele.predict(x_test)

    accuracy = accuracy_score(y_test, y_predit)
    precision = precision_score(y_test, y_predit, pos_label="Filière Scientifique")
    recall = recall_score(y_test, y_predit, pos_label="Filière Scientifique")
    f1 = f1_score(y_test, y_predit, pos_label="Filière Scientifique")

    # Matrice de confusion (on fixe l'ordre des classes pour que ce soit lisible)
    classes = ["Filière Scientifique", "Filière Littéraire"]
    matrice = confusion_matrix(y_test, y_predit, labels=classes)

    # Prédiction pour le nouvel élève saisi dans le formulaire
    nouvel_eleve = [[demande.note, demande.heures]]
    orientation_predite = modele.predict(nouvel_eleve)[0]

    return {
        "orientation_predite": orientation_predite,
        "accuracy": vider_valeurs_numpy(round(accuracy, 3)),
        "precision": vider_valeurs_numpy(round(precision, 3)),
        "recall": vider_valeurs_numpy(round(recall, 3)),
        "f1_score": vider_valeurs_numpy(round(f1, 3)),
        "matrice_confusion": vider_valeurs_numpy(matrice),
        "classes": classes,
        "message": "Rappel : la décision finale d'orientation appartient toujours au conseil de classe.",
    }


# =====================================================================
# 10. ROUTE RACINE (juste pour vérifier que le serveur fonctionne)
# =====================================================================

@app.get("/")
def accueil():
    return {"message": "API du TP INF232 - Lycée de MBOUDA. Le serveur fonctionne correctement."}
