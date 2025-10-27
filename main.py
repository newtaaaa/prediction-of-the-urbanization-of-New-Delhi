"""
Thème de l'année : Transition, transformation, conversion.
/main.py version définitive/

L'ordre suivant est celui des données de couverture terrestre :

    Zone bâtie → Classe 1
    Végétation → Classe 2
    Plan d'eau → Classe 3
    Autres → Classe 4
"""

import os
import numpy as np
from osgeo import gdal
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import math


def distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def barycentre(points):
    if not points:
        return []
    d = len(points[0])
    result = [0.0] * d
    for point in points:
        for i in range(d):
            result[i] += point[i]
    return [x / len(points) for x in result]


def k_moyenne(E, k, max_iter=100):
    mu = random.sample(E, k)
    clusters = []
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for x in E:
            index_min = 0
            dist_min = distance(x, mu[0])
            for j in range(1, k):
                d = distance(x, mu[j])
                if d < dist_min:
                    dist_min = d
                    index_min = j
            clusters[index_min].append(x)
        new_mu = []
        for cluster in clusters:
            new_mu.append(barycentre(cluster))
        if all(distance(mu[i], new_mu[i]) < 1e-6 for i in range(k)):
            break
        mu = new_mu
    return clusters


# Définition de la fonction pour lire un fichier raster et retourner un tableau et la source de données
def readraster(file):
    dataSource = gdal.Open(file)
    print(dataSource.GetRasterBand)
    band = dataSource.GetRasterBand(1)
    band = band.ReadAsArray()
    return dataSource, band


def identicalList(inList):
    global logical
    inList = np.array(inList)
    logical = inList == inList[0]
    return sum(logical) == len(inList)


def difference_periode(terrain1, terrain2, buclass=1, cellsize=30):
    return (sum(sum(((terrain2 == buclass).astype(int) - (terrain1 == buclass).astype(int)) != 0)) * (
            cellsize ** 2) / 1000000)


# Définition d'une classe pour lire les fichiers de couverture terrestre de deux périodes temporelles

class Landcover:
    def __init__(self, file1, file2, file3):
        self.matriceNorm = None
        self.matrice = None
        self.ds_lc1, self.arr_lc1 = readraster(file1)  # fichier gdal, tableau
        # des cellules bâties
        self.ds_lc2, self.arr_lc2 = readraster(file2)
        self.ds_lc3, self.arr_lc3 = readraster(file3)
        self.nClasses = 4  # nombre de classes de cellules possibles
        self.taille_correspondance()  # méthode de verification de la correspondance de
        # la taille des entrées

    def taille_correspondance(self):
        # Vérification des dimensions des rasters en entrée
        print("Vérification de la taille des rasters en entrée...")
        if (self.ds_lc1.RasterXSize == self.ds_lc2.RasterXSize) and (
                self.ds_lc1.RasterYSize == self.ds_lc2.RasterYSize):
            print("Les tailles des données de couverture terrestre correspondent.")
            self.ligne, self.col = (self.ds_lc1.RasterYSize, self.ds_lc1.RasterXSize)
        else:
            print("Les fichiers de couverture terrestre en entrée ont des hauteurs et largeurs différentes.")
        # Vérification du nombre de classes dans les images de couverture terrestre
        print("\nVérification des classes d'occupation du sol...")
        if (self.arr_lc1.max() == self.arr_lc2.max()) and (self.arr_lc1.min() == self.arr_lc2.min()):
            print("Les classes des fichiers de couverture terrestre en entrée correspondent.")
            self.nClasses = len(np.unique(self.arr_lc1))  # nb classes distinctes
        else:
            print("Les données de couverture terrestre en entrée ont des valeurs de classe différentes.")

    def matrice_transition(self):
        n_classes = max(self.arr_lc1.max(), self.arr_lc2.max())  # nombre total de classes
        self.matrice = np.zeros((n_classes, n_classes), dtype=int)
        for x in range(self.ligne):
            for y in range(self.col):
                t1_pixel = self.arr_lc1[x, y]
                t2_pixel = self.arr_lc2[x, y]
                if t1_pixel > 0 and t2_pixel > 0:  # Ignore les NoData (valeurs 0)
                    self.matrice[t1_pixel - 1, t2_pixel - 1] += 1
        print("\nMatrice de transition calculée. Normalisation...")
        self.matriceNorm = np.zeros_like(self.matrice, dtype=float)
        for i in range(self.matrice.shape[0]):
            rangee_somme = self.matrice[i, :].sum()
            if rangee_somme > 0:
                self.matriceNorm[i, :] = self.matrice[i, :] / rangee_somme
            else:
                self.matriceNorm[i, :] = 0


class FacteursCroissance:  # facteurs de croissance
    def __init__(self, *args):
        self.gf = dict()  # dictionnaire de matrices, chaque pixel
        # contenu correspond à une valeur propre au facteur considéré
        self.gf_ds = dict()  # dictionnaire des fichiers gdal des facteurs
        self.nfacteurs = len(args)  # nombre de facteurs
        self.n = 1
        for fichier in args:  # remplir les dictionnaires
            self.gf_ds[self.n], self.gf[self.n] = readraster(fichier)
            self.n += 1
        self.taille_correspondance()

    def taille_correspondance(self):
        print("\nVérification de la taille des facteurs de croissance en entrée...")
        lignes = []
        cols = []
        for n in range(1, self.nfacteurs + 1):
            lignes.append(self.gf_ds[n].RasterYSize)
            cols.append(self.gf_ds[n].RasterXSize)
        if identicalList(lignes) and identicalList(cols):
            print("Les facteurs en entrée ont le même nombre de lignes et de colonnes.")
            self.ligne = self.gf_ds[self.nfacteurs].RasterYSize
            self.col = self.gf_ds[self.nfacteurs].RasterXSize
        else:
            print("Les facteurs en entrée ont des dimensions différentes.")


class Fitmodel:
    def __init__(self, landcoverClass: Landcover, facteursClass: FacteursCroissance):
        self.spatialAccuracy = 0
        self.construction_fictive = 0
        self.construction_reelle = 0
        self.seuils = []
        self.seuil_construction = 0
        self.prediction = []  # tableau de cellules bâties
        self.landcovers = landcoverClass
        self.facteurs = facteursClass  # facteurs de croissance
        self.taille_correspondance()
        self.noyau = 3

    def taille_correspondance(self):
        print("\nCorrespondance de la taille de la couverture terrestre et des facteurs de croissance...")
        if (self.landcovers.ligne == self.facteurs.ligne) and (self.landcovers.col == self.facteurs.col):
            print("Taille des rasters correspondante.")
            self.rangee = self.facteurs.ligne
            self.col = self.facteurs.col
        else:
            print("ERREUR ! La taille des rasters ne correspond pas, veuillez vérifier.")

    def setseuils(self, seuil_construction, *OtherseuilssInSequence):
        self.seuils = list(OtherseuilssInSequence)
        self.seuil_construction = seuil_construction
        if len(self.seuils) == (len(self.facteurs.gf)):
            print("\nSeuil défini pour les facteurs")
        else:
            print("ERREUR ! Veuillez vérifier le nombre de facteurs.")

    def predire(self):
        self.prediction = deepcopy(self.landcovers.arr_lc2)
        marge_lat = math.ceil(self.noyau / 2)
        transition_probs = self.landcovers.matriceNorm  # Matrice de transition
        for y in range(marge_lat, self.rangee - (marge_lat - 1)):
            for x in range(marge_lat, self.col - (marge_lat - 1)):
                kernel = self.landcovers.arr_lc1[y - (marge_lat - 1):y + marge_lat,
                         x - (marge_lat - 1):x + marge_lat]
                compteur_construction = np.sum(kernel == 1)  # nombre de bâties
                classe_act = self.landcovers.arr_lc2[y, x]
                if classe_act < 1 or classe_act > transition_probs.shape[0]:
                    continue  # ignorer des valeurs innatendues
                proba_construire = transition_probs[classe_act - 1, 0]  # [classe][se_construire]
                if (compteur_construction >= self.seuil_construction) and (self.facteurs.gf[5][y, x] != 1):
                    score = 0
                    for facteur in range(1, self.facteurs.nfacteurs + 1):
                        val = self.facteurs.gf[facteur][y, x]
                        seuil = self.seuils[facteur - 1]
                        if seuil < 0 and val <= abs(seuil):
                            score += 1
                        elif 0 < seuil <= val:
                            score += 1
                    # on combine le score et la probabilité de construction
                    if score >= 3 and proba_construire >= 0.25:  # le seuil de référence est arbitraire.
                        self.prediction[y, x] = 1
                if (y % 500 == 0) and (x % 500 == 0):
                    print("rangee: %d, Col: %d, Builtup cells count: %d\n" % (y, x, compteur_construction), end="\r",
                          flush=True)
    
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
            self.exactitude()

            if self.spatialAccuracy > best_accuracy:
                best_accuracy = self.spatialAccuracy
                best_k = k
                best_prediction = deepcopy(self.prediction)
        # Réappliquer la meilleure config
        print(f"\n✅ Meilleur k trouvé : {best_k} avec une exactitude de {best_accuracy:.2f}%")

        self.predire_kmoyenne(n_clusters=best_k)
        self.prediction = best_prediction
        self.exactitude()
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

    def exactitude(self):
        # Exactitude statistique
        self.construction_reelle = difference_periode(self.landcovers.arr_lc2, self.landcovers.arr_lc3)
        self.construction_fictive = difference_periode(self.landcovers.arr_lc2, self.prediction)

        # Calcul de l'exactitude spatiale
        self.spatialAccuracy = 100 - (
                sum(sum(((self.prediction == 1).astype(float) - (self.landcovers.arr_lc2 == 1).astype(float)) != 0)) /
                sum(sum(self.landcovers.arr_lc2 == 1))
        ) * 100

        print("Croissance réelle : %d, Croissance prédite : %d" % (self.construction_reelle, self.construction_fictive))

        # Affichage de l'exactitude spatiale
        print("Exactitude spatiale : %f" % self.spatialAccuracy)

    def exportprediction(self, outFileName):
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outFileName, self.col, self.rangee, 1, gdal.GDT_UInt16)  # option : GDT_UInt16, GDT_Float32
        outdata.SetGeoTransform(self.landcovers.ds_lc1.GetGeoTransform())
        outdata.SetProjection(self.landcovers.ds_lc1.GetProjection())
        outdata.GetRasterBand(1).WriteArray(self.prediction)
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()
        outdata = None


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

file3 = "Actual_1999.tif"

cbd = "cbddist.tif"
road = "roaddist.tif"
restricted = "dda_2021_government_restricted.tif"
pop01 = "den1991.tif"
pop11 = "den2011.tif"
pop19 = "den2019.tif"
pop24 = "den2024.tif"
slope = "slope.tif"
ds = gdal.Open("slope.tif")

file1 = "Actual_1989.tif"
file2 = "Actual_1994.tif"

couvertureTerrain = Landcover(file1, file2, file3)
facteurs = FacteursCroissance(cbd, road, pop01, slope, restricted)
prediction = Fitmodel(couvertureTerrain, facteurs)
prediction.setseuils(3, -15000, -10000, 8000, -3, -1)

'''
caModel.find_best_k()
afficher_clusters(caModel)
'''

# Exécuter le modèle
prediction.predire()

# Vérifier l'exactitude des valeurs prédites
prediction.exactitude()
# Exporter la couche prédite
prediction.exportprediction('Nouveau_rapport_1999.tif')
