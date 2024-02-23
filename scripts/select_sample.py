import os
import random
import shutil

def select_images(folder, number_samples, path_destiny):
    files_images = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                files_images.append(os.path.join(root, filename))

    samples_selected = random.sample(files_images, number_samples)

    if not os.path.exists(path_destiny):
        os.makedirs(path_destiny)

    for image in samples_selected:
        name_folder = os.path.basename(os.path.dirname(image))
        name_file = os.path.basename(image)
        name_file_destiny = f"{name_folder}_{name_file}"
        name_file_destiny = name_file_destiny.lower()
        shutil.copy(image, os.path.join(path_destiny, name_file_destiny))

    print("Se han seleccionado y copiado", number_samples, "imágenes al directorio de destino.")

folder = '/home/satoru/repos/u_andes/maia/mlns'
number_samples = 100
path_destiny = '/home/satoru/repos/u_andes/maia/mlns/micro_projects/one/sample'

select_images(folder, number_samples, path_destiny)
