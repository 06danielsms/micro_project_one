import streamlit as st
from PIL import Image
import numpy as np
from logic.image_process import ImageProcess

image_process = ImageProcess()

# Título de la aplicación
st.title('Aplicación de Procesamiento de Imágenes')

# Sidebar para cargar la imagen y ajustar los parámetros
st.sidebar.title('Configuración')
st.set_option('deprecation.showPyplotGlobalUse', False)
uploaded_image = st.sidebar.file_uploader("Cargar imagen", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Mostrar la imagen cargada
    st.sidebar.image(uploaded_image, caption='Imagen Original', use_column_width=True)

    # Ajustar parámetros
    target_size = st.sidebar.slider("Seleccionar tamaño (píxeles)", 50, 500, 200)
    transform = st.sidebar.checkbox('Transform')
    # Convertir la imagen cargada a un formato que se pueda procesar
    image = Image.open(uploaded_image)
    file_name = uploaded_image.name
    image = image_process.load_single_image(image, file_name)
    title = image[0]['name']
    # Procesar la imagen con los parámetros seleccionados
    image_result, images_parcial = image_process.pre_process_single_image(image, transform=transform, target_size=target_size)
    
    #image = Image.fromarray(imagen_procesada[0])

    # Mostrar la imagen procesada
    st.image(images_parcial[0], caption=f'Imagen Procesada: {title}', use_column_width=True)
    centroids, labels = image_process.kmeans_image(image_result[0], num_clusters=7)
    image_process.plot_colors(centroids)
    x_train_reduced = image_process.dimensionality_reduction(image_result[0], centroids[labels], st_plot=True)        
    elbow = image_process.elbow_method(x_train_reduced, st_plot=True)
    silhouette = image_process.silhouette_plot(x_train_reduced, st_plot=True)
    
    