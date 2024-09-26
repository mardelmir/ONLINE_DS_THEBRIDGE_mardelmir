import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples


# KMeans

X = 'array'
k = 5
kmeans = KMeans(n_clusters = k)

kmeans_per_k = [KMeans(n_clusters = k, random_state = 42, n_init = 25).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

# HAcer función aquí
plt.figure(figsize = (8, 3.5))
plt.plot(range(1, 10), inertias, 'bo-')
plt.xlabel('$k$', fontsize = 14)
plt.ylabel('Inertia', fontsize = 14)
plt.annotate('Elbow/Codo',
             xy = (4, inertias[3]),
             xytext = (0.55, 0.55),
             textcoords = 'figure fraction',
             fontsize = 16,
             arrowprops = dict(facecolor = 'black', shrink = 0.1)
            )
plt.axis([1, 8.5, 0, 1300])
plt.show()

# Hacer función aquí
'''
La anchura de cada cuchillo representa el numero de muestras por cluster. 
Están ordenadas por su coeficiente de silhouette, por eso tiene esa forma de cuchillo. 
Cuanta más caida tenga indica que las muestras tienen un coeficiente mas disperso en ese cluster
Deberian estar todos los clusters por encima de la media.
Hay algunas líneas hacia la izda porque es el coeficiente negativo. Puntos asignados al cluster erroneo.
'''
plt.figure(figsize = (11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)
    
    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor = color, edgecolor = color, alpha = 0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel('Cluster')
    
    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel('Silhouette Coefficient')
    else:
        plt.tick_params(labelbottom = True)

    plt.axvline(x = silhouette_scores[k - 2], color = 'red', linestyle = '--')
    plt.title('$k={}$'.format(k), fontsize = 16)

plt.show()

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize = 2)


def plot_centroids(centroids, weights = None, circle_color = 'w', cross_color = 'b'):
    if weights is not None:
        centroids  =  centroids[weights > weights.max() / 10]
        
    plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'o', s = 30, linewidths = 8, color = circle_color, zorder = 10, alpha = 0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 15, linewidths = 20, color = cross_color, zorder = 11, alpha = 1)


def plot_decision_boundaries(clusterer, X, resolution = 1000, show_centroids = True, show_xlabels = True, show_ylabels = True):
    mins = X.min(axis = 0) - 0.1
    maxs = X.max(axis = 0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent = (mins[0], maxs[0], mins[1], maxs[1]), cmap = 'Pastel2')
    plt.contour(Z, extent = (mins[0], maxs[0], mins[1], maxs[1]), linewidths = 1, colors = 'k')
    plot_data(X)
    
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel('$x_1$', fontsize = 14)
    else:
        plt.tick_params(labelbottom = False)
        
    if show_ylabels:
        plt.ylabel('$x_2$', fontsize = 14, rotation = 0)
    else:
        plt.tick_params(labelleft = False)
        
        
# DBSCAN

def plot_dbscan(dbscan, X, size, show_xlabels = True, show_ylabels = True):
    core_mask = np.zeros_like(dbscan.labels_, dtype = bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1], c = dbscan.labels_[core_mask], marker = 'o', s = size, cmap = 'Paired')
    plt.scatter(cores[:, 0], cores[:, 1], marker = '*', s = 20, c = dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c = 'r', marker = 'x', s = 100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c = dbscan.labels_[non_core_mask], marker = '.')
    
    if show_xlabels:
        plt.xlabel('$x_1$', fontsize = 14)
    else:
        plt.tick_params(labelbottom = False)
        
    if show_ylabels:
        plt.ylabel('$x_2$', fontsize = 14, rotation = 0)
    else:
        plt.tick_params(labelleft = False)
        
    plt.title('eps = {:.2f}, min_samples = {}'.format(dbscan.eps, dbscan.min_samples), fontsize = 14)