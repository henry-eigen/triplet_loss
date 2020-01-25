import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import os

from keras.utils import to_categorical

import psutil

def throttle_cpu(count):
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in range(count)])


def load_dataset(path, num_per_class, verbose=False, shuffle=True):
    img_dict = {}
    labels = []

    dims = (224, 224)
    
    #path = "/data1/Henry/train/"

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #for idx, class_name in enumerate(os.listdir(path))

    for idx, class_name in enumerate(classes):
        if verbose:
            print(class_name)
        img_dict[str(idx)] = []
        class_path = os.path.join(path, class_name)
        for i in range(num_per_class):
            im = Image.open(os.path.join(class_path, os.listdir(class_path)[i]))
            im = im.resize(dims, Image.ANTIALIAS)
            labels.append(idx)
            img_dict[str(idx)].append(np.array(im))

    imgs = np.array([img_dict[i] for i in img_dict.keys()]).reshape((len(labels), 224, 224, 3))

    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=10)
    #labels = labels - labels * 0.05

    if shuffle:
        p = np.random.permutation(len(labels))

        imgs = imgs[p]

        labels = labels[p]
    
    return (imgs, labels)