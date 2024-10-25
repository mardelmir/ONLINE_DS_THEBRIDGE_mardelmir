import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.io import imread
import zipfile


def unzip_file(zip_path, extract_to_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f'Successfully extracted {zip_path} to {extract_to_path}')


def load_images_from_folder(folder, label, reshape_dim = (32,32)):
    X = []
    y = []
    
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder, filename))
            img = img.resize(reshape_dim)
            
            if img is not None:
                X.append(np.array(img))
                y.append(filename.split('.')[0])
            
    return np.array(X), np.array(y)


def show_images_batch(imgs, labels, class_names, n_cols = 5, cmap = plt.cm.binary):
    n_rows = (len(imgs) - 1) // n_cols + 1
    plt.figure(figsize = (n_cols * 1.75, n_rows * 1.75))
    
    for i in range(len(imgs)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(class_names[labels[i]])
        plt.imshow(imgs[i], cmap = cmap)
        plt.axis('off')
    plt.show()
    
    
def plot_image(i, img, class_predicted, class_prob, true_label):
  img = img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap = plt.cm.binary)
  plt.xlabel(f'{class_predicted} {class_prob:.2f} ({true_label})')

def plot_value_array(predictions_array, predictions_labels):
  #predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(len(predictions_labels)),labels = predictions_labels)
  plt.yticks([])
  thisplot = plt.bar(range(len(predictions_array)), predictions_array, color = '#777777')
  plt.ylim([0, 1])
  plt.xlabel('top-5 predicciones')

  #thisplot[predicted_label].set_color('red')
  #thisplot[true_label].set_color('blue')