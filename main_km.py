"""
Ce module est conçu pour simuler la croissance urbaine
à l'aide de données de couverture terrestre, de paramètres
de croissance urbaine et d'un modèle d'automates cellulaires.

L'ordre suivant est celui des données de couverture terrestre :
    Zone bâtie --> Classe 1
    Végétation --> Classe 2
    Plan d'eau --> Classe 3
    Autres --> Classe 4
"""

import os
import numpy as np
from osgeo import gdal
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import random
import math


def distance(a, b):
    """Calcule la distance euclidienne entre deux points a et b."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def mean(points):
    """Calcule le barycentre (moyenne) d'une liste de points."""
    if not points:
        return []
    d = len(points[0])
    result = [0.0] * d
    for point in points:
        for i in range(d):
            result[i] += point[i]
    return [x / len(points) for x in result]


def k_means(E, k, max_iter=100):
    """Algorithme k-moyennes sans numpy, version lisible."""
    # Étape 1 : Initialisation aléatoire des k centres
    mu = random.sample(E, k)
    clusters = []
    for _ in range(max_iter):
        # Étape 2 : Initialiser les k clusters vides
        clusters = [[] for _ in range(k)]
        # Assigner chaque point au cluster le plus proche
        for x in E:
            index_min = 0
            dist_min = distance(x, mu[0])
            for j in range(1, k):
                d = distance(x, mu[j])
                if d < dist_min:
                    dist_min = d
                    index_min = j
            clusters[index_min].append(x)
        # Recalculer les barycentres
        new_mu = []
        for cluster in clusters:
            new_mu.append(mean(cluster))
        # Vérifier si les barycentres ont changé
        if all(distance(mu[i], new_mu[i]) < 1e-6 for i in range(k)):
            break  # Convergence atteinte
        mu = new_mu  # Mettre à jour les centres
    return clusters


# Exemple d'utilisation
if __name__ == "__main__":
    E = [
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ]
    k = 2
    resultats = k_means(E, k)
    for i, cluster in enumerate(resultats):
        print(f"Cluster {i + 1} : {cluster}")


# Définition de la fonction pour lire un fichier raster et retourner un tableau et la source de données
def readraster(file):
    dataSource = gdal.Open(file)
    print(dataSource.GetRasterBand)
    band = dataSource.GetRasterBand(1)
    band = band.ReadAsArray()
    return (dataSource, band)


def identicalList(inList):
    global logical
    inList = np.array(inList)
    logical = inList == inList[0]
    return sum(logical) == len(inList)


def builtupAreaDifference(landcover1, landcover2, buclass=1, cellsize=30):
    return (sum(sum(((landcover2 == buclass).astype(int) - (landcover1 == buclass).astype(int)) != 0)) * (
            cellsize ** 2) / 1000000)


# Définition d'une classe pour lire les fichiers de couverture terrestre de deux périodes temporelles


class landcover():
    def __init__(self, file1, file2, file3):
        self.ds_lc1, self.arr_lc1 = readraster(file1)
        self.ds_lc2, self.arr_lc2 = readraster(file2)
        self.ds_lc3, self.arr_lc3 = readraster(file3)
        self.npropriete = None
        self.performChecks()

    def performChecks(self):
        # Vérification des dimensions des rasters en entrée
        print("Vérification de la taille des rasters en entrée...")
        if (self.ds_lc1.RasterXSize == self.ds_lc2.RasterXSize) and (
                self.ds_lc1.RasterYSize == self.ds_lc2.RasterYSize):
            print("Les tailles des données de couverture terrestre correspondent.")
            self.rangee, self.col = (self.ds_lc1.RasterYSize, self.ds_lc1.RasterXSize)
        else:
            print("Les fichiers de couverture terrestre en entrée ont des hauteurs et largeurs différentes.")
        # Vérification du nombre de classes dans les images de couverture terrestre
        print("\nVérification des classes d'occupation du sol...")
        if (self.arr_lc1.max() == self.arr_lc2.max()) and (self.arr_lc1.min() == self.arr_lc2.min()):
            print("Les classes des fichiers de couverture terrestre en entrée correspondent.")
            self.npropriete = len(np.unique(self.arr_lc1))  # nb classes distinctes
        else:
            print("Les données de couverture terrestre en entrée ont des valeurs de classe différentes.")

    def transitionMatrix(self):
        self.tMatrix = np.random.randint(1, size=(self.npropriete, self.npropriete))
        for x in range(0, self.rangee):
            for y in range(0, self.col):
                t1_pixel = self.arr_lc1[x, y]
                t2_pixel = self.arr_lc2[x, y]
                self.tMatrix[t1_pixel - 1, t2_pixel - 1] += 1
        self.tMatrixNorm = np.random.randint(1, size=(self.npropriete, self.npropriete)).astype(float)
        print("\nMatrice de transition calculée, normalisation en cours...")
        # Création de la matrice de transition normalisée
        for x in range(0, self.tMatrix.shape[0]):
            for y in range(0, self.tMatrix.shape[1]):
                self.tMatrixNorm[x, y] = self.tMatrix[x, y] / (self.tMatrix[x, :]).sum()


class grangeethfacteurs():
    def __init__(self, *args):
        self.gf = dict()
        self.gf_ds = dict()
        self.nfacteurs = len(args)
        n = 1
        for file in args:
            self.gf_ds[n], self.gf[n] = readraster(file)
            n += 1
        self.performChecks()

    def performChecks(self):
        print("\nVérification de la taille des facteurs de croissance en entrée...")
        rangees = []
        cols = []
        for n in range(1, self.nfacteurs + 1):
            rangees.append(self.gf_ds[n].RasterYSize)
            cols.append(self.gf_ds[n].RasterXSize)
        if identicalList(rangees) and identicalList(cols):
            print("Les facteurs en entrée ont le même nombre de lignes et de colonnes.")
            self.rangee = self.gf_ds[n].RasterYSize
            self.col = self.gf_ds[n].RasterXSize
        else:
            print("Les facteurs en entrée ont des dimensions différentes.")


class fitmodel():

    def __init__(self, landcoverClass, grangeethfacteursClass):
        self.clustered = []
        self.prediction = []
        self.landcovers = landcoverClass
        self.facteurs = grangeethfacteursClass
        self.performChecks()
        self.kernelSize = 3
        self.accuracy = 0

    def performChecks(self):
        print("\nCorrespondance de la taille de la couverture terrestre et des facteurs de croissance...")
        if (self.landcovers.rangee == self.facteurs.rangee) and (self.landcovers.col == self.facteurs.col):
            print("Taille des rasters correspondante.")
            self.rangee = self.facteurs.rangee
            self.col = self.facteurs.col
        else:
            print("ERREUR ! La taille des rasters ne correspond pas, veuillez vérifier.")

    def setThreshold(self, builtupThreshold, *OtherThresholdsInSequence):
        self.threshold = list(OtherThresholdsInSequence)
        self.builtupThreshold = builtupThreshold
        if len(self.threshold) == (len(self.facteurs.gf)):
            print("\nSeuil défini pour les facteurs")
        else:
            print("ERREUR ! Veuillez vérifier le nombre de facteurs.")

    def predict(self):
        self.prediction = deepcopy(self.landcovers.arr_lc1)
        sideMargin = math.ceil(self.kernelSize / 2)
        for y in range(sideMargin, self.rangee - (sideMargin - 1)):
            for x in range(sideMargin, self.col - (sideMargin - 1)):
                kernel = self.landcovers.arr_lc1[y - (sideMargin - 1):y + (sideMargin),
                         x - (sideMargin - 1):x + (sideMargin)]
                builtupCount = sum(sum(kernel == 1))
                # Si le nombre de cellules bâties est supérieur ou égal au seuil attribué
                if (builtupCount >= self.builtupThreshold) and (
                        self.facteurs.gf[5][y, x] != 1):  # Ajout de l'exception pour les zones restreintes
                    for factor in range(1, self.facteurs.nfacteurs + 1):
                        # Si les seuils attribués sont inférieurs à zéro, alors la règle "moins que" s'applique, sinon "plus grand que"
                        if self.threshold[factor - 1] < 0:
                            if (self.facteurs.gf[factor][y, x] <= abs(self.threshold[factor - 1])):
                                self.prediction[y, x] = 1
                            else:
                                pass
                        elif self.threshold[factor - 1] > 0:
                            if (self.facteurs.gf[factor][y, x] >= self.threshold[factor - 1]):
                                self.prediction[y, x] = 1
                            else:
                                pass
                if (y % 500 == 0) and (x % 500 == 0):
                    print("Ligne : %d, Colonne : %d, Nombre de cellules bâties : %d\n" % (y, x, builtupCount), end="\r",
                          flush=True)

    def checkAccuracy(self):
        # Exactitude statistique
        self.actualBuildup = builtupAreaDifference(self.landcovers.arr_lc2, self.landcovers.arr_lc3)
        self.predictionBuildup = builtupAreaDifference(self.landcovers.arr_lc2, self.prediction)

        # Calcul de l'exactitude spatiale
        self.accuracy = 100 - (
                sum(sum(((self.prediction == 1).astype(float) - (self.landcovers.arr_lc3 == 1).astype(float)) != 0)) /
                sum(sum(self.landcovers.arr_lc3 == 1))
        ) * 100

        print("Croissance réelle : %d, Croissance prédite : %d" % (self.actualBuildup, self.predictionBuildup))

        # Affichage de l'exactitude spatiale
        print("Exactitude spatiale : %f" % (self.accuracy))

    def exportprediction(self, outFileName):
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outFileName, self.col, self.rangee, 1,
                                gdal.GDT_UInt16)  # option : GDT_UInt16, GDT_Float32
        outdata.SetGeoTransform(self.landcovers.ds_lc1.GetGeoTransform())
        outdata.SetProjection(self.landcovers.ds_lc1.GetProjection())
        outdata.GetRasterBand(1).WriteArray(self.prediction)
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()
        outdata = None

    def predire_kmoyenne(self, n_clusters, ratio_max_pixels_per_cluster=0.2, voisinage_seuil=3):
        print("Clustering des données avec K-means...")

        # Créer un masque pour le territoire (exclure le fond)
        territoire_mask = (self.landcovers.arr_lc2 != 0)
        # un pixel de fond a pour valeur 0
        proprietes = []
        coords = []  # pour garder la position des pixels valides

        for x in range(self.rangee):
            for y in range(self.col):
                if not territoire_mask[x, y]:
                    continue  # Ignorer les pixels hors territoire
                vecteur = [self.facteurs.gf[i][x, y] for i in range(1, self.facteurs.nfacteurs + 1)]
                proprietes.append(vecteur)  # matrice de taille Npixels * Nfacteurs
                coords.append((x, y))

        proprietes = np.array(proprietes)

        # Standardisation
        scaler = StandardScaler()
        proprietes_scaled = scaler.fit_transform(proprietes)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(proprietes_scaled)

        # Créer une matrice clusterisée initialisée à -1 (hors territoire)
        self.clustered = -1 * np.ones((self.rangee, self.col), dtype=int)

        # Remplir la matrice clusterisée seulement aux indices valides
        for (y, x), label in zip(coords, labels):
            self.clustered[y, x] = label

        # Analyse du taux de bâtis par cluster
        cluster_stats = dict()
        for c in range(n_clusters):
            mask = (self.clustered == c)
            if mask.sum() == 0:
                cluster_stats[c] = 0
                continue
            builtup_ratio = np.sum((self.landcovers.arr_lc2 == 1) & mask) / np.sum(mask)
            cluster_stats[c] = builtup_ratio

        for c, ratio in cluster_stats.items():
            print(f"Cluster {c} : ratio de bâtis = {ratio:.2f}")

        # ratio de bâtis enregistré dans un dictionnaire cluster_stats
        seuil_auto = 0.125 * max(cluster_stats.values())
        propices = [c for (c, v) in cluster_stats.items() if v >= seuil_auto]
        print(f"Clusters propices identifiés (seuil {seuil_auto:.2f}) : {propices}")

        # Prédiction avec règles de voisinage et limitation
        self.prediction = deepcopy(self.landcovers.arr_lc2)
        for c in propices:
            mask = (self.clustered == c) & (self.landcovers.arr_lc2 != 1)
            indices = list(zip(*np.where(mask)))
            random.shuffle(indices)
            limit = int(ratio_max_pixels_per_cluster * len(indices))

            changed = 0
            for x, y in indices:
                if 1 <= x < self.rangee - 1 and 1 <= y < self.col - 1:
                    kernel = self.landcovers.arr_lc2[x - 1:x + 2, y - 1:y + 2]
                    voisins_constr = np.sum(kernel == 1)
                    if voisins_constr >= voisinage_seuil:
                        self.prediction[x, y] = 1
                        changed += 1
                if changed >= limit:
                    break

        print("Croissance prédite terminée.")

    def find_best_k(self, k_min=5, k_max=5):
        best_k = None
        best_accuracy = -1
        best_prediction = None

        for k in range(k_min, k_max + 1):
            self.prediction = deepcopy(self.landcovers.arr_lc2)
            print(f"\n🔄 Test avec k = {k}")
            self.predire_kmoyenne(n_clusters=k)
            self.checkAccuracy()

            if self.accuracy > best_accuracy:
                best_accuracy = self.accuracy
                best_k = k
                best_prediction = deepcopy(self.prediction)
        # Réappliquer la meilleure config
        print(f"\n✅ Meilleur k trouvé : {best_k} avec une exactitude de {best_accuracy:.2f}%")

        self.predire_kmoyenne(n_clusters=best_k)
        self.prediction = best_prediction
        self.checkAccuracy()
        self.exportprediction("meilleur_v2.tif")

    def methode_coude(self, k_min=1, k_max=10):
        print("Calcul des inerties pour la méthode du coude...")
        territoire_mask = (self.landcovers.arr_lc2 != 0)
        proprietes = []

        for y in range(self.rangee):
            for x in range(self.col):
                if not territoire_mask[y, x]:
                    continue
                vecteur = [self.facteurs.gf[i][y, x] for i in range(1, self.facteurs.nfacteurs + 1)]
                proprietes.append(vecteur)

        proprietes = np.array(proprietes)
        scaler = StandardScaler()
        proprietes_scaled = scaler.fit_transform(proprietes)

        inertias = []
        K = range(k_min, k_max + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
            kmeans.fit(proprietes_scaled)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(K, inertias, 'o-', linewidth=2)
        plt.xlabel("Nombre de clusters (k)")
        plt.ylabel("Inertie (Within-Cluster Sum of Squares)")
        plt.title("Méthode du coude pour choisir k")
        plt.grid(True)
        plt.show()


##################################################
## Le modèle d'automate cellulaire se termine   ##
## La partie ci-dessous du code doit être mise à jour ##
##################################################

# Attribuer le répertoire où se trouvent les fichiers


# Nombre de bandes

def afficher_clusters(caModel):
    plt.figure(figsize=(10, 8))
    cluster_map = np.copy(caModel.clustered).astype(float)
    cluster_map[cluster_map == -1] = np.nan  # Pixels hors territoire transparents
    plt.imshow(cluster_map, cmap='tab10')
    plt.title("proprietete des clusters K-means")
    plt.colorbar(label="Cluster ID")
    plt.axis("off")
    plt.show()


os.chdir(r"C:\Users\natha\OneDrive\Documents\TIPE")

# Entrée des fichiers GeoTIFF pour deux périodes de temps
file1 = "Actual_1989.tif"
file2 = "Actual_1994.tif"
file3 = "Actual_1999.tif"

# Entrée de tous les paramètres
cbd = "cbddist.tif"
road = "roaddist.tif"
restricted = "dda_2021_government_restricted.tif"
pop01 = "den1991.tif"
pop11 = "den2011.tif"
pop19 = "den2019.tif"
pop24 = "den2024.tif"
slope = "slope.tif"
ds = gdal.Open("slope.tif")

# Créer une classe de couverture terrestre qui prend les données de couverture terrestre pour deux périodes
myLandcover = landcover(file1, file2, file3)

# Créer une classe de facteurs qui configure tous les facteurs pour le modèle
myfacteurs = grangeethfacteurs(cbd, road, pop01, slope, restricted)

# Initialiser le modèle avec les classes créées ci-dessus
caModel = fitmodel(myLandcover, myfacteurs)

# Selon l'exactitude statistique et spatiale affichée, les seuils doivent être ajustés
caModel.setThreshold(3, -15000, -10000, 8000, -3, -1)

# Exécuter le modèle
caModel.find_best_k()
afficher_clusters(caModel)

'''
caModel.checkAccuracy()
caModel.exportprediction("prediction_bestK.tif")'''

# Exporter la couche prédite
caModel.exportprediction('Nouveau_rapport.tif')
