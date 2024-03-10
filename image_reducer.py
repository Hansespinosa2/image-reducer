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

def crop_and_cut(file_path,crop_size,n_clusters):
    an_image = plt.imread(file_path)
    if n_clusters is not None:        
        an_image = segment_colors(an_image,n_clusters)

    an_image = tf.keras.layers.Resizing(crop_size,crop_size,crop_to_aspect_ratio=True)(an_image)
    return np.array(an_image)

def make_images(file_path_list:list,crop_size,color_restriction):
    return [crop_and_cut(path,crop_size,color_restriction) for path in file_path_list]


def display_and_save_images(output,path,crop_size):
    for i,img in enumerate(output):
        plt.imshow(np.array(img))
        plt.axis('off')
        # plt.savefig(path + '/' + f'{i}.png',transparent=True,dpi=crop_size)
        plt.imsave(fname=path + '/' + f'{i}.png', arr=np.array(img), format='png')

    




def reduce_image_folder(folder:str, crop_size:int=640, pool:int=4,color_restriction=None):
    """
    Reduces the size of images in a folder by performing resizing, center cropping and max pooling.
    Place any images of desired size in a folder and call this function to reduce the size of the images.
    By default, the image will be pixellated and reduced to a 160x160 image.
    Using crop_size of 320 and a pool of 4 will reduce the image to 80x80.

    Args:
        folder (str): The path to the folder containing the images.
        crop_size (int, optional): The size of the smallest crop to be performed on the images. Defaults to 640.
        pool (int, optional): The size of the max pooling window. Defaults to 4.
        color_restriction (int, optional): The maximum number of colors allowed in the reduced images. Defaults to 10.
    """

    the_file_paths = get_file_paths(folder)
    the_images = make_images(the_file_paths,crop_size,color_restriction)
    the_images = tf.keras.layers.CenterCrop(height=crop_size,width=crop_size)(the_images)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=pool)

    output = max_pool(the_images)

    if not os.path.exists(folder + '_reduced'):
        os.makedirs(folder + '_reduced')

    display_and_save_images(output,folder + '_reduced',crop_size)





    
    