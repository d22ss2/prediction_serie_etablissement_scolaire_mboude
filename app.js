/* =====================================================================
   app.js
   Toute la logique côté frontend : navigation entre onglets,
   appels fetch() vers l'API FastAPI, et affichage des graphiques
   avec Chart.js.
   Le code est volontairement simple : uniquement des fonctions,
   pas de classes ni de framework.
   ===================================================================== */

// Adresse de base de l'API (le backend FastAPI doit tourner sur ce port)
const URL_API = "http://127.0.0.1:8000";

// On garde en mémoire les graphiques déjà créés, pour pouvoir les
// détruire avant d'en recréer un nouveau (sinon Chart.js superpose
// les anciens graphiques).
let graphiqueHistogramme = null;
let graphiqueBoxplot = null;
let graphiqueNuage = null;
let graphiqueCluster = null;

// On garde aussi en mémoire l'équation de régression, pour le simulateur
let penteRegression = 0;
let ordonneeRegression = 0;
let rCarreRegression = 0;


/* =====================================================================
   NAVIGATION ENTRE LES ONGLETS
   ===================================================================== */

function changerOnglet(nomOnglet) {
    // On cache tous les onglets
    document.querySelectorAll(".onglet").forEach(function (onglet) {
        onglet.classList.remove("visible");
    });

    // On enlève la classe "actif" de tous les boutons du menu
    document.querySelectorAll(".bouton-onglet").forEach(function (bouton) {
        bouton.classList.remove("actif");
    });

    // On affiche l'onglet demandé
    document.getElementById("onglet-" + nomOnglet).classList.add("visible");

    // On met en évidence le bouton cliqué
    event.target.classList.add("actif");
}


/* =====================================================================
   ACCUEIL : GENERATION DES DONNEES
   ===================================================================== */

async function genererDonnees() {
    const nom = document.getElementById("champ-nom").value;

    if (nom.trim() === "") {
        alert("Veuillez saisir le nom complet du chef de groupe.");
        return;
    }

    const reponse = await fetch(URL_API + "/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nom: nom }),
    });

    const donnees = await reponse.json();

    // Affichage de la seed et du nombre d'élèves
    document.getElementById("resultat-generation").innerHTML =
        "<div class='resultat-encadre'>Seed calculée : " + donnees.seed +
        " — " + donnees.nombre_eleves + " élèves générés avec succès.</div>";

    // Construction du tableau (aperçu des 200 élèves)
    afficherTableauEleves(donnees.donnees);
}


function afficherTableauEleves(listeEleves) {
    let html = "<tr><th>ID</th><th>Note Math</th><th>Heures de travail</th><th>Orientation</th></tr>";

    listeEleves.forEach(function (eleve) {
        html += "<tr>" +
            "<td>" + eleve.id + "</td>" +
            "<td>" + eleve.note_math + "</td>" +
            "<td>" + eleve.heures_travail + "</td>" +
            "<td>" + eleve.orientation + "</td>" +
            "</tr>";
    });

    document.getElementById("tableau-eleves").innerHTML = html;
}


/* =====================================================================
   ONGLET 1 : ANALYSE DESCRIPTIVE
   ===================================================================== */

async function chargerAnalyseDescriptive() {
    const reponse = await fetch(URL_API + "/api/statistics/univariate");
    const resultat = await reponse.json();

    if (resultat.erreur) {
        alert(resultat.erreur);
        return;
    }

    // Affichage des cartes de statistiques
    const cartes = [
        { titre: "Moyenne", valeur: resultat.moyenne },
        { titre: "Médiane", valeur: resultat.mediane },
        { titre: "Variance", valeur: resultat.variance },
        { titre: "Écart-type", valeur: resultat.ecart_type },
        { titre: "Minimum", valeur: resultat.minimum },
        { titre: "Maximum", valeur: resultat.maximum },
        { titre: "Q1", valeur: resultat.q1 },
        { titre: "Q3", valeur: resultat.q3 },
    ];

    let htmlCartes = "";
    cartes.forEach(function (carte) {
        htmlCartes += "<div class='carte'><div>" + carte.titre + "</div>" +
            "<div class='valeur'>" + carte.valeur + "</div></div>";
    });
    document.getElementById("cartes-statistiques").innerHTML = htmlCartes;

    // Interprétation automatique
    document.getElementById("interpretation-descriptive").innerText =
        "Interprétation : " + resultat.interpretation;

    // ----- Histogramme -----
    const bornes = resultat.histogramme.bornes;
    const effectifs = resultat.histogramme.effectifs;

    // On crée des étiquettes du type "0-2", "2-4", etc.
    const etiquettesHistogramme = [];
    for (let i = 0; i < effectifs.length; i++) {
        etiquettesHistogramme.push(bornes[i] + " à " + bornes[i + 1]);
    }

    if (graphiqueHistogramme !== null) {
        graphiqueHistogramme.destroy();
    }

    const contexteHistogramme = document.getElementById("graphique-histogramme").getContext("2d");
    graphiqueHistogramme = new Chart(contexteHistogramme, {
        type: "bar",
        data: {
            labels: etiquettesHistogramme,
            datasets: [{
                label: "Nombre d'élèves",
                data: effectifs,
                backgroundColor: "#3873ad",
            }],
        },
        options: {
            plugins: { title: { display: true, text: "Distribution des notes" } },
        },
    });

    // ----- Boxplot (représentation simple avec un graphique en barres horizontales) -----
    // Chart.js ne propose pas de boxplot natif : on affiche donc les
    // 5 valeurs clés (min, Q1, médiane, Q3, max) sous forme de barres.
    const boxplot = resultat.boxplot;

    if (graphiqueBoxplot !== null) {
        graphiqueBoxplot.destroy();
    }

    const contexteBoxplot = document.getElementById("graphique-boxplot").getContext("2d");
    graphiqueBoxplot = new Chart(contexteBoxplot, {
        type: "bar",
        data: {
            labels: ["Minimum", "Q1", "Médiane", "Q3", "Maximum"],
            datasets: [{
                label: "Note",
                data: [boxplot.min, boxplot.q1, boxplot.mediane, boxplot.q3, boxplot.max],
                backgroundColor: ["#a8d5b0", "#3873ad", "#1e3a5f", "#3873ad", "#a8d5b0"],
            }],
        },
        options: {
            indexAxis: "y",
            plugins: { title: { display: true, text: "Résumé en 5 nombres (approximation du boxplot)" } },
        },
    });
}


/* =====================================================================
   ONGLET 2 : ANALYSE BIVARIEE
   ===================================================================== */

async function chargerAnalyseBivariee() {
    const reponse = await fetch(URL_API + "/api/statistics/bivariate");
    const resultat = await reponse.json();

    if (resultat.erreur) {
        alert(resultat.erreur);
        return;
    }

    // On garde les coefficients de régression pour le simulateur
    penteRegression = resultat.pente;
    ordonneeRegression = resultat.ordonnee_origine;
    rCarreRegression = resultat.r_carre;

    // Affichage des cartes
    const cartes = [
        { titre: "Covariance", valeur: resultat.covariance },
        { titre: "Corrélation", valeur: resultat.correlation },
        { titre: "R²", valeur: resultat.r_carre },
        { titre: "Équation", valeur: resultat.equation },
    ];

    let htmlCartes = "";
    cartes.forEach(function (carte) {
        htmlCartes += "<div class='carte'><div>" + carte.titre + "</div>" +
            "<div class='valeur' style='font-size:14px'>" + carte.valeur + "</div></div>";
    });
    document.getElementById("cartes-bivariee").innerHTML = htmlCartes;

    // ----- Nuage de points + droite de régression -----
    const points = resultat.points;

    // On calcule deux points pour tracer la droite de régression
    // (une droite est définie par deux points minimum)
    const xMin = Math.min(...points.map(function (p) { return p.x; }));
    const xMax = Math.max(...points.map(function (p) { return p.x; }));
    const droiteRegression = [
        { x: xMin, y: penteRegression * xMin + ordonneeRegression },
        { x: xMax, y: penteRegression * xMax + ordonneeRegression },
    ];

    if (graphiqueNuage !== null) {
        graphiqueNuage.destroy();
    }

    const contexteNuage = document.getElementById("graphique-nuage").getContext("2d");
    graphiqueNuage = new Chart(contexteNuage, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Élèves",
                    data: points,
                    backgroundColor: "#3873ad",
                },
                {
                    label: "Droite de régression",
                    data: droiteRegression,
                    type: "line",
                    borderColor: "#e74c3c",
                    borderWidth: 2,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            scales: {
                x: { title: { display: true, text: "Heures de travail" } },
                y: { title: { display: true, text: "Note de mathématiques" } },
            },
        },
    });
}


function simulerNote() {
    const heures = parseFloat(document.getElementById("champ-heures-simulateur").value);

    if (isNaN(heures)) {
        alert("Veuillez saisir un nombre d'heures valide.");
        return;
    }

    if (penteRegression === 0 && ordonneeRegression === 0) {
        alert("Veuillez d'abord cliquer sur 'Calculer la relation'.");
        return;
    }

    const noteEstimee = penteRegression * heures + ordonneeRegression;

    let htmlResultat = "<div class='resultat-encadre'>Pour " + heures +
        " heures de travail, la note estimée est : " + noteEstimee.toFixed(2) + "/20</div>";

    // Si R² est faible, on affiche un avertissement
    if (rCarreRegression < 0.5) {
        htmlResultat += "<div class='avertissement'>⚠️ Attention : le coefficient R² est faible (" +
            rCarreRegression + "). Cette prédiction est peu fiable.</div>";
    }

    document.getElementById("resultat-simulateur").innerHTML = htmlResultat;
}


/* =====================================================================
   ONGLET 3 : CLASSIFICATION NON SUPERVISEE
   ===================================================================== */

async function chargerClustering() {
    const reponse = await fetch(URL_API + "/api/cluster", { method: "POST" });
    const resultat = await reponse.json();

    if (resultat.erreur) {
        alert(resultat.erreur);
        return;
    }

    document.getElementById("info-clustering").innerText =
        "Nombre de groupes trouvés : " + resultat.nombre_groupes +
        " (score de silhouette : " + resultat.score_silhouette + ")";

    // On sépare les points par groupe pour leur donner une couleur différente
    const couleurs = ["#3873ad", "#e74c3c", "#2ecc71", "#f39c12"];
    const datasets = [];

    for (let g = 0; g < resultat.nombre_groupes; g++) {
        const pointsDuGroupe = resultat.points.filter(function (p) { return p.groupe === g; });
        datasets.push({
            label: "Groupe " + (g + 1),
            data: pointsDuGroupe.map(function (p) { return { x: p.x, y: p.y }; }),
            backgroundColor: couleurs[g % couleurs.length],
        });
    }

    // On ajoute les centres des clusters comme un dataset à part
    datasets.push({
        label: "Centres des clusters",
        data: resultat.centres.map(function (c) { return { x: c.heures_travail, y: c.note_math }; }),
        backgroundColor: "black",
        pointStyle: "cross",
        radius: 10,
        borderWidth: 3,
    });

    if (graphiqueCluster !== null) {
        graphiqueCluster.destroy();
    }

    const contexteCluster = document.getElementById("graphique-cluster").getContext("2d");
    graphiqueCluster = new Chart(contexteCluster, {
        type: "scatter",
        data: { datasets: datasets },
        options: {
            scales: {
                x: { title: { display: true, text: "Heures de travail" } },
                y: { title: { display: true, text: "Note de mathématiques" } },
            },
        },
    });

    // Description textuelle de chaque groupe
    let htmlDescriptions = "";
    for (const numeroGroupe in resultat.descriptions) {
        htmlDescriptions += "<p><strong>Groupe " + (parseInt(numeroGroupe) + 1) + "</strong> : " +
            resultat.descriptions[numeroGroupe] + "</p>";
    }
    document.getElementById("description-groupes").innerHTML = htmlDescriptions;
}


/* =====================================================================
   ONGLET 4 : CLASSIFICATION SUPERVISEE
   ===================================================================== */

async function predireOrientation() {
    const note = parseFloat(document.getElementById("champ-note-predire").value);
    const heures = parseFloat(document.getElementById("champ-heures-predire").value);

    if (isNaN(note) || isNaN(heures)) {
        alert("Veuillez remplir la note et les heures de travail.");
        return;
    }

    const reponse = await fetch(URL_API + "/api/classify/train-and-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note: note, heures: heures }),
    });

    const resultat = await reponse.json();

    if (resultat.erreur) {
        alert(resultat.erreur);
        return;
    }

    // Affichage du résultat de la prédiction
    document.getElementById("resultat-prediction").innerHTML =
        "<div class='resultat-encadre'>Orientation prédite : " + resultat.orientation_predite + "</div>" +
        "<p><em>" + resultat.message + "</em></p>";

    // Affichage des indicateurs de performance
    const cartes = [
        { titre: "Accuracy", valeur: resultat.accuracy },
        { titre: "Precision", valeur: resultat.precision },
        { titre: "Recall", valeur: resultat.recall },
        { titre: "F1-score", valeur: resultat.f1_score },
    ];

    let htmlCartes = "";
    cartes.forEach(function (carte) {
        htmlCartes += "<div class='carte'><div>" + carte.titre + "</div>" +
            "<div class='valeur'>" + carte.valeur + "</div></div>";
    });
    document.getElementById("cartes-performance").innerHTML = htmlCartes;

    // Affichage de la matrice de confusion sous forme de tableau
    const matrice = resultat.matrice_confusion;
    const classes = resultat.classes;

    let htmlMatrice = "<tr><th></th><th>Prédit : " + classes[0] + "</th><th>Prédit : " + classes[1] + "</th></tr>";
    htmlMatrice += "<tr><th>Réel : " + classes[0] + "</th><td>" + matrice[0][0] + "</td><td>" + matrice[0][1] + "</td></tr>";
    htmlMatrice += "<tr><th>Réel : " + classes[1] + "</th><td>" + matrice[1][0] + "</td><td>" + matrice[1][1] + "</td></tr>";

    document.getElementById("tableau-matrice").innerHTML = htmlMatrice;
}
