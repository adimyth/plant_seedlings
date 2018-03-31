import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16,10]
plt.rcParams['font.size'] = 16
from mpl_toolkits.axes_grid1 import ImageGrid
import glob
import os
import cv2
import numpy as np

num_classes = 5
base_dir = "data/validation/"
images = []
categories = os.listdir(base_dir)
for category in categories:
    path = os.path.join(base_dir, category)
    total = glob.glob(os.path.join(path, "*.png"))
    images.append(total[:5])
images = list(np.array(images).flatten())
# print(images)

fig = plt.figure(1, figsize=(num_classes, num_classes))
grid = ImageGrid(fig, 111, nrows_ncols=(num_classes, num_classes), axes_pad=0.05)
i=0
for index, category in enumerate(categories):
    start = index*num_classes
    end = (index+1)*num_classes
    for file in images[start:end]:
        # print(file)
        ax = grid[i]
        img = cv2.imread(file)
        img = cv2.resize(img, dsize=(224, 224))
        ax.imshow(img/255.)
        ax.axis('off')
        if i%num_classes == num_classes-1:
            ax.text(250, 112, category, verticalalignment='center')
        i+=1
plt.show()

import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

IMG_WIDTH = 45
IMG_HEIGHT = 45
num_classes = 12
train_dir = os.path.join("data", "train")
imgs = []
labels = []

print("Reading Images...")
files = glob(os.path.join(train_dir, "*/*.png"))
for file in files:
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.flatten()
    imgs.append(img)
imgs = np.asarray(imgs)

print("Generating Classes...")
for index,folder in enumerate(os.listdir(train_dir)):
    path = os.path.join(train_dir, folder, "*.png")
    for images in glob(path):
        labels.append(folder)

labels = np.array(labels)
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])

pca = PCA(n_components=180)
pca_result = pca.fit_transform(imgs)
print(pca_result.shape, type(pca_result))

tsne = TSNE(n_components=2, perplexity=40.0)
tnse_result = tsne.fit_transform(pca_result)
print(tnse_result.shape, type(tnse_result))

def visualize_scatter(data_2d, label_ids, figsize=(20,20)):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.show()
visualize_scatter(imgs, label_ids)