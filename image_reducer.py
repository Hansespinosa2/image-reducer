import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_file_paths(folder:str):
    for root, _, file_paths in os.walk(folder):
        the_file_paths = [os.path.join(root,path) for path in file_paths]
    return the_file_paths

def crop_and_cut(file_path,smallest_crop):
    an_image = plt.imread(file_path)
    length = an_image.shape[0]
    width = an_image.shape[1]
    if length < smallest_crop or width < smallest_crop:
        print(f'{file_path} is smaller than the crop size, skipping')
        exit
    start_l = (length - smallest_crop)//2
    start_w = (width-smallest_crop)//2
    an_image = an_image[start_l:smallest_crop+start_l,start_w:smallest_crop+start_w,:3]
    return an_image

def make_images(file_path_list:list,smallest_crop):
    return [crop_and_cut(path,smallest_crop) for path in file_path_list]


def display_and_save_images(output,path):
    for i,img in enumerate(output):
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.savefig(path + '/' + f'{i}.png')

    

def reduce_image_folder(folder:str, smallest_crop:int=320, pool:int=4):
    the_file_paths = get_file_paths(folder)
    the_images = make_images(the_file_paths,smallest_crop)
    the_images = tf.keras.layers.CenterCrop(height=smallest_crop,width=smallest_crop)(the_images)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=pool)

    output = max_pool(the_images)

    if not os.path.exists(folder + '_reduced'):
        os.makedirs(folder + '_reduced')

    display_and_save_images(output,folder + '_reduced')





    
    