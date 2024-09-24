import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

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