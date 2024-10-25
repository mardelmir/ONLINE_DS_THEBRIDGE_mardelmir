import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
import zipfile


def unzip_file(zip_path, extract_to_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f'Successful extraction of {zip_path} to {extract_to_path}')
    

def read_image_data(directory, reshape_dim = (32,32)):
    X = []
    y = []
    for folder in os.listdir(directory):
        if os.path.isdir('/'.join([directory, folder])):
            for file in os.listdir('/'.join([directory, folder])):

                image = imread('/'.join([directory, folder, file]))
                image = cv2.resize(image, reshape_dim)

                X.append(image)
                y.append(folder)

    return np.array(X),np.array(y)


def show_images_batch(images, names = [], n_cols = 5, size_scale = 2, cmap = plt.cm.binary):
    n_rows = ((len(images) - 1) // n_cols + 1)
    plt.figure(figsize = (n_cols * size_scale, n_rows * 1.1 * size_scale))
    
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image, cmap = cmap)
        plt.axis('off')
        if len(names):
            plt.title(names[i])