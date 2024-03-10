import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def get_file_paths(folder:str):
    for root, _, file_paths in os.walk(folder):
        the_file_paths = [os.path.join(root,path) for path in file_paths]
    return the_file_paths


def segment_colors(an_image,n_clusters):
    original_shape = an_image.shape
    X = np.array(an_image).reshape(-1, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img.reshape(original_shape)

def crop_and_cut(file_path,smallest_crop,n_clusters):
    an_image = plt.imread(file_path)
    if n_clusters is not None:        
        an_image = segment_colors(an_image,n_clusters)

    an_image = tf.keras.layers.Resizing(smallest_crop,smallest_crop,crop_to_aspect_ratio=True)(an_image)
    return np.array(an_image)

def make_images(file_path_list:list,smallest_crop,color_restriction):
    return [crop_and_cut(path,smallest_crop,color_restriction) for path in file_path_list]


def display_and_save_images(output,path):
    for i,img in enumerate(output):
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.savefig(path + '/' + f'{i}.png',transparent=True)

    




def reduce_image_folder(folder:str, smallest_crop:int=320, pool:int=4,color_restriction=10):
    the_file_paths = get_file_paths(folder)
    the_images = make_images(the_file_paths,smallest_crop,color_restriction)
    the_images = tf.keras.layers.CenterCrop(height=smallest_crop,width=smallest_crop)(the_images)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=pool)

    output = max_pool(the_images)

    if not os.path.exists(folder + '_reduced'):
        os.makedirs(folder + '_reduced')

    display_and_save_images(output,folder + '_reduced')





    
    