import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import re
import pandas as pd
import random
import webcolors
from PIL import Image
import streamlit as st

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

class ImageProcess:
    def __init__(self, root_path=None):
        self.root_path = root_path
        if not root_path is None:
            self.images_original = self.load_images(root_path)
            self.images_process = self.pre_process_image(self.images_original, flatten=True, normalize=True, target_size=100)
    
    def load_images(self, root_path):
        files = os.listdir(root_path)
        random.shuffle(files)
        images_original = []
        for file in files:
            img = cv2.imread(os.path.join(root_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_original.append({'image': img, 'name': file})
        return images_original

    def load_single_image(self, image, file_name):
        images_original = []
        image = np.array(image)
        images_original.append({'image': image, 'name': file_name})
        return images_original
    
    
    def pre_process_single_image(self, images_original, transform=False, flatten=True, normalize=True, target_size=None):
        images_process = []
        images_parcial = []
        for data in images_original:
            image = data['image']
            if transform:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not target_size is None:
                height, width = data['image'].shape[:2]
                ratio = float(target_size) / max(height, width)
                new_height = int(height * ratio)
                new_width = int(width * ratio)
                image = cv2.resize(image, (new_width, new_height))
            images_parcial.append(image)
            if flatten:
                image = image.reshape(-1, 3)
            if normalize:
                image = image / 255.0
            images_process.append(image)
        return images_process, images_parcial
    
    def pre_process_image(self, images_original, flatten=False, normalize=False, target_size=None):
        images_process = []
        for data in images_original:
            height, width = data['image'].shape[:2]
            if target_size is not None:
                ratio = float(target_size) / max(height, width)
                new_height = int(height * ratio)
                new_width = int(width * ratio)
                image = cv2.resize(data['image'], (new_width,new_height))
            if flatten:
                image = image.reshape((-1, 3))
            if normalize:
                image = image / 255.0
            images_process.append(image)
        return images_process
    
    def kmeans_image(self, img, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(img)
        return kmeans.cluster_centers_, kmeans.labels_
    
    def dimensionality_reduction(self, img, colors, tsne=False, p=3, st_plot=False):
        if not tsne:
            pca = PCA(n_components=2)
            x_train_reduced = pca.fit_transform(img)
        else:
            tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=1)
            x_train_reduced = tsne.fit_transform(img)

        colors = colors * 255
        colors = list(map(self.rgb_to_hex, colors))
        color_dict = {}
        for i in colors:
            color_dict[i] = i
        colors = np.array(colors)

        for category in np.unique(colors):
            mask = colors == category
            plt.scatter(x_train_reduced[:,0][mask], x_train_reduced[:,1][mask], label=category, color=color_dict[category], edgecolors='black')

        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.title('Dispersión PCA colores')
        plt.legend()
        if st_plot:
            st.pyplot()
        else:
            plt.show()
        return x_train_reduced

    def rgb_to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def rgb_to_color_hex(self, rgb_tuple):
        try:
            color_name = webcolors.rgb_to_hex(rgb_tuple)
        except ValueError:
            color_name = 'Unknown'
        return color_name

    def draw_image_palette(self, image_original, centroids=None, st_plot=False):
        palette = centroids * 255
        title = image_original['name'][:-3]
        title = re.sub(r'[^A-Za-z0-9\-]+', ' ', title).upper()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"{title}")
        plt.imshow(image_original['image'])
        plt.axis('off')

        if centroids is not None:
            plt.subplot(1, 2, 2)
            plt.title('Palette')
            palette_colors = []
            color_names = []
            for color in palette:
                palette_colors.append(color)
                color_name = self.rgb_to_color_hex(tuple(color.astype(int)))
                color_names.append(color_name)
            plt.imshow(np.expand_dims(palette_colors, axis=0).astype(np.uint8))
            plt.xticks([])
            plt.yticks([])
            for i, name in enumerate(color_names):
                plt.text(i, .5, name, ha='center', va='top', color='black', fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        if st_plot:
            st.pyplot()
        else:
            plt.show()
    
    def plot_colors(self, colors):
        plt.figure(figsize=(8, 6))
        for i in range(len(colors)):
            color_swatch = np.zeros((100, 100, 3))
            color_swatch[:, :, :] = colors[i]
            plt.subplot(1, len(colors), i + 1)
            plt.imshow(color_swatch)
            plt.axis('off')
        st.pyplot()
    

    def elbow_method(self, data, max_clusters=10, st_plot=False):
        distortions = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        # Plotting the elbow method graph
        plt.plot(range(1, max_clusters + 1), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        if st_plot:
            st.pyplot()
        else:
            plt.show()


    def silhouette_analysis(self, data, max_clusters=10, st_plot=False):
        silhouette_scores = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        # Plotting silhouette scores
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        if st_plot:
            st.pyplot()
        else:
            plt.show()


    def silhouette_plot(self, data, max_clusters=10, st_plot=False):
        scores = []
        for i in range(2, max_clusters + 1):
            model_k = KMeans(n_clusters=i, n_init=10, random_state=42)
            # Entrenamos el modelo
            model_k.fit(data)
            # Almacenamos el coeficiente de la silueta
            score = silhouette_score(data, model_k.labels_)
            scores.append(score)
        # Mostramos los valores de los coeficientes
        print(pd.DataFrame({'K': range(2, max_clusters+1), 'Coeficiente': scores}))
        # Graficamos los valores del coeficiente de la silueta
        plt.plot(range(2, max_clusters+1), scores, marker='o')
        plt.xlabel('Número de clústeres')
        plt.ylabel('Silhouette Score')
        plt.grid()
        if st_plot:
            st.pyplot()
        else:
            plt.show()
    
    def analyze_images(self, num_clusters=7, num_images=2):
        processed_images = 0
        for i, image in enumerate(self.images_process):
            print(f"****************************************{i}************************************")
            centroids, labels = self.kmeans_image(self.images_process[i], num_clusters)
            self.draw_image_palette(self.images_original[i], centroids)
            x_train_reduced = self.dimensionality_reduction(self.images_process[i], centroids[labels])        
            self.elbow_method(x_train_reduced)
            self.silhouette_plot(x_train_reduced)
            processed_images += 1
            if processed_images >= num_images:
                break
            
if __name__ == "__main__":
    root_path = '/home/satoru/repos/u_andes/maia/mlns/micro_projects/one/sample'
    image_analyzer = ImageProcess(root_path)
    image_analyzer.analyze_images()
