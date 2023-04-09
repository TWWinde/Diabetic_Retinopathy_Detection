import csv
import os
import shutil
import cv2
import gin
import numpy as np
import pandas as pd
from input_pipeline.preprocessing import augment, Ben_preprocess_circle


def get_image_names_labels(path):
    labels = pd.read_csv(path)
    labels = np.asarray(labels)
    return labels


# Using this function to write binary labels and get oversampled, preprocessed and augmented image data
@gin.configurable
def processing_augmentation_oversampling(lb_path, save_path, img_path, amount):  # train=True
    # if test dataset; train==0 without augmentation # amount :the wanted number of images per class
    # multiplier of the number of pictures in training set
    def setDir(filepath):
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            shutil.rmtree(filepath, ignore_errors=True)

    setDir(save_path + 'image/')
    setDir(save_path + 'image/' + 'train/')
    setDir(save_path + 'image/' + 'test/')
    title = ['name', 'label']
    with open(save_path + 'test.csv', 'a', newline='', encoding='UTF-8') as f2:
        writer = csv.writer(f2)
        writer.writerow(title)
    label_imagename = get_image_names_labels(lb_path + 'test.csv')
    for i in range(len(label_imagename)):
        image = cv2.imread(img_path + 'test/' + label_imagename[i, 0] + '.jpg')
        if label_imagename[i, 1] <= 1:
            add = [label_imagename[i, 0], 0]
            with open(save_path + 'test.csv', 'a', newline='', encoding='UTF-8') as f2:
                writer = csv.writer(f2)
                writer.writerow(add)
            image = Ben_preprocess_circle(image)
            cv2.imwrite(save_path + 'image/test/' + label_imagename[i, 0] + ".jpg", image * 255)
            print('saving image', label_imagename[i, 0])
        else:
            add = [label_imagename[i, 0], 1]
            with open(save_path + 'test.csv', 'a', newline='', encoding='UTF-8') as f2:
                writer = csv.writer(f2)
                writer.writerow(add)
            image = Ben_preprocess_circle(image)
            cv2.imwrite(save_path + 'image/test/' + label_imagename[i, 0] + ".jpg", image * 255)
            print('saving image', label_imagename[i, 0])
    with open(save_path + 'train.csv', 'a', newline='', encoding='UTF-8') as f2:
        writer = csv.writer(f2)
        writer.writerow(title)
    label_imagename = get_image_names_labels(lb_path + 'train.csv')
    k = 1
    count0 = 0
    count1 = 0
    i = 0
    print('...........................')
    while (count1 < amount) or (count0 < amount):
        if (label_imagename[i, 1] <= 1) & (count0 < amount):
            image = cv2.imread(img_path + 'train/' + label_imagename[i, 0] + '.jpg')
            if k > 413:
                image = augment(image)  # augumentation
            image = Ben_preprocess_circle(image)
            image = np.asarray(image)
            a = str(k).zfill(3)
            cv2.imwrite(save_path + 'image/train/' + "IDRiD_" + a + ".jpg", image * 255)
            print('saving image', "IDRiD_", a)
            add = ['IDRiD_' + a, 0]
            with open(save_path + 'train.csv', 'a', newline='', encoding='UTF-8') as f2:
                writer = csv.writer(f2)
                writer.writerow(add)
            count0 += 1
            i += 1
            k += 1
            if i  == 413:
                i = 0
        elif (label_imagename[i, 1] > 1) & (count1 < amount):
            image = cv2.imread(img_path + 'train/' + label_imagename[i, 0] + '.jpg')
            if k > 413:
                image = augment(image)  # augumentation
            image = Ben_preprocess_circle(image)
            image = np.asarray(image)
            a = str(k).zfill(3)
            cv2.imwrite(save_path + 'image/train/' + "IDRiD_" + a + ".jpg", image * 255)
            print('saving image', "IDRiD_", a)
            add = ['IDRiD_' + a, 1]
            with open(save_path + 'train.csv', 'a', newline='', encoding='UTF-8') as f2:
                writer = csv.writer(f2)
                writer.writerow(add)
            count1 += 1
            i += 1
            k += 1
            if i == 413:
                i = 0
        else:
            i += 1
            if i == 413:
                i = 0