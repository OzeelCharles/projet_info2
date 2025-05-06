# 📁 Projet informatique: Étude des résultats aux grand prix de F1 depuis 2009

## 🏁 Résumé du projet

Ce projet a pour objectif d'explorer et d’analyser diverses caractéristiques de la Formule 1 à l’aide d’outils Python. En répondant à une série de questions ciblées, il met en œuvre des techniques d’analyse de données, de visualisation et de modélisation statistique pour extraire des insights pertinents sur les pilotes, les écuries et les courses.

Les analyses incluent notamment :

- **Q1** : Comment se répartie les circuits de la formule 1 sur la terre ?
- **Q2** : Comment comparer deux pilotes selons leurs résultats de courses ?
- **Q3** : Comment se répartie le temps d'arrêt au stand des pilotes dont on possède les données ?
- **Q4** : 
- **Q5** : 
- **Q6** : 
- **Q7** : 
- **Q9** : 
- **Q10** : 


Ce projet utilise principalement `pandas`, `numpy`, `matplotlib`, `scikit-learn` et `geopandas`, et s’exécute dans un environnement Jupyter Notebook ou VSCode.

---

## 🧾 Sommaire

- [Contexte](#contexte)
- [Fonctionnalités](#fonctionnalités)
- [Technologies utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Contributeurs](#contributeurs)
- [Licence](#licence)

---

## 🧐 Contexte

## 🏎️ Contexte

La Formule 1 est l'une des disciplines sportives les plus médiatisées au monde et l'une des plus exigente. Chaque saison, des dizaines de courses sont organisées à travers le globe, mettant en compétition les meilleurs pilotes et les écuries les plus prestigieuses. Mais au-delà du spectacle, la F1 est aussi une incroyable source de données. Résultats de courses, temps au tour, positions sur la grille, vitesses moyennes, performances en qualifications, stratégies d’arrêts aux stands, conditions météorologiques, abandons... la richesse et la variété des données disponibles sont immenses.

Toutefois, cette abondance d'informations présente aussi un défi : comment structurer, filtrer et analyser efficacement ces données pour en extraire du sens ? Les données sont hétérogènes (numériques, catégorielles, géospatiales, temporelles), avec du bruit, incomplètes. 

L’objectif de ce projet est justement d’apporter de la clarté dans cet pagaille. En répondant à une série de questions analytiques, nous cherchons à départager les pilotes, comprendre les dynamiques entre saisons, identifier les facteurs de performance et tenter de découvrir les variables qui pourraient permettre de prédire de futures victoires. La complexité du sport, mêlant performances humaines, techniques, aléas, rend cette tâche aussi fascinante que difficile — mais elle ouvre également la porte à une exploration analytique passionnante.

---

## ✨ Fonctionnalités

Listez les principales fonctionnalités du projet :
- ✅ Importation des données à l'aide d'un fichier contenant l'ensemble des bases de données
- ✅ Représentation graphique de différentes statistiques
- ✅ Estimation du nombre de points par nombre de victoire
- ✅ Estimation du nombre de points par durée moyen au stand
- ✅ statitique moyenne du temps passé au stand par joueur
- ✅ Nombre de points sur toute la carrière du pilote 
- ✅ Nombre de victoires sur toute la carrière du pilote
- ✅ Visualisaiton sur une des positions des circuits de grand prix
- ✅ taux de victoire total par pilote
- ✅ 

---

### 📊 Source des données

Les données utilisées dans ce projet ont été collectées à partir de **Wikipedia**, qui propose une base d'informations riche et détaillée sur l’histoire de la Formule 1. Elles comprennent notamment :

- Les résultats de courses par saison
- Les classements pilotes et écuries
- Les vitesses moyennes, dates, circuits, et abandons
- Les données complémentaires sur les Grands Prix (emplacements, horaires, etc.)

Ces données sont sous forme de fichiers CSV.

À noter également :

- Le fichier **`naturalearth_lowres`** provient des **données intégrées à la bibliothèque [GeoPandas](https://geopandas.org/en/stable/docs/reference/api/geopandas.datasets.get_path.html)**. Il s’agit d’un jeu de données géographiques mondial en basse résolution, fourni par le projet [Natural Earth](https://www.naturalearthdata.com/), utilisé ici pour afficher les cartes de fond dans les visualisations spatiales.

--- 

## 🛠 Technologies utilisées

- **Langage principal** : Python 3.11  
- **Bibliothèques** :
  - `pandas` – manipulation de données  
  - `numpy` – calcul numérique  
  - `matplotlib` – visualisation  
  - `scikit-learn` – machine learning  
  - `geopandas` – traitement et visualisation de données géospatiales  
- **Environnement** : Jupyter Notebook / VSCode  

---

## ⚙️ Installation

```bash
# Cloner le projet
git clone https://github.com/OzeelCharles/projet_info2.git

# Aller dans le dossier
#depuis windows:
C:\Users\nom_utilisateur\Desktop\projet_info2
#depuis macOS ou Linux: 
/home/nom_utilisateur/Desktop/projet_info2

#Avant d'exécuter ce projet, assurez-vous d'avoir installé les dépendances nécessaires. C'est à dire avoir sur votre bureau donnees_formule_un, d'avoir sur le même dépot que le fichier jupyter_final.ipynb, le fichier naturalearth_lowres, et f1_grands_prix_localisation.ipynb

pip install -r requirements.txt

## 👥 Auteurs

- Charles OZEEL – interface GIT, Module Charles.py, README.md, Notebook_final.ipynb – [@Lachance#233020](https://github.com/OzeelCharles)
- Jules – – []()
- Lilian - -[]()
- Gabriel - -[]()
