import matplotlib.pyplot as plt
import scipy.misc as sm
from imageio import imread
import numpy as np
import cv2
import os
import sys
from code_material_python.helper import *
from code_material_python.hfs import *

from sklearn.preprocessing import normalize

def _brightness(img, value=100):
    if value > 0:
        shadow = value
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + value
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow
    return cv2.addWeighted(img, alpha_b, img, 0, gamma_b)


def _face_detection(im, frame_size, cc, brightness=-1,
                    filter_method="Gaussian"):

    box = cc.detectMultiScale(im)
    top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}
    for cfx, cfy, clx, cly in box:
        face_area = clx * cly
        if face_area > top_face["area"]:
            top_face["area"] = face_area
            top_face["box"] = [cfx, cfy, clx, cly]

    x, y, w, h = top_face["box"]
    face = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face = face[y: y+h, x: x+w]
    face = cv2.resize(face,(frame_size, frame_size), interpolation = cv2.INTER_CUBIC)
    face = face.astype(np.uint8)

    if brightness > 0:
        face = _brightness(face, value=brightness)

    if filter_method == "box":
        face = cv2.boxFilter(face, -1, (5, 5))
    elif filter_method == "bilinear":
        face = cv2.bilateralFilter(face, 9, 75, 75)
    elif filter_method == "Gaussian":
        face = cv2.GaussianBlur(face, (5, 5), 0)

    face = cv2.equalizeHist(face)
    return face



def _random_masked_labels(n_pers, n_im, n_label=1):
    """
    >>> _random_masked_labels(4, 4, 2)
    array([1., 0., 1., 0., 2., 2., 0., 0., 3., 3., 0., 0., 0., 4., 0., 4.])"""
    labels = np.ones((n_pers*n_im))
    for i in range(n_pers):
        labels[i*n_im:(i+1)*n_im] = i+1
        masked = np.random.choice(n_im, n_im-n_label, replace=False) + i*n_im
        labels[masked] = 0
    return labels


def _fetch_dataset(data='code_material_python/data/10faces/', \
                    classifier='code_material_python/data/haarcascade_frontalface_default.xml', \
                    frame_size=96, brightness=-1, extended=False, \
                    n_pers=2, n_im=2, filter_method="Gaussian"):
    """ Return the images and labels from the dataset """
    images = np.zeros((n_pers*n_im, frame_size ** 2))
    labels = np.zeros(n_pers*n_im, dtype=int)
    digits = 3 if extended else 2
    extension = "" if extended else ".jpg"
    cc = cv2.CascadeClassifier(classifier)

    for i in np.arange(n_pers):
        for j in np.arange(n_im):

            im = imread(os.path.join(data, str(i), str(j+1).zfill(digits)  + extension))
            gray_face = _face_detection(im, frame_size, cc, brightness=brightness)
            images[i * n_im + j] = gray_face.reshape((-1))
            labels[i * n_im + j] = i + 1

    return images, labels


def _plot_dataset(images, labels, n_pers, n_im):
    plt.figure(1)
    for i in range(n_pers*n_im):
        plt.subplot(n_pers, n_im,i+1)
        plt.axis('off')
        plt.imshow(images[i].reshape(frame_size,frame_size), cmap="gray")
        if i % n_im == 0:
            plt.title(f'Person {labels[i]}')
    plt.show()


def _visualization_face_detection(labels, Y_masked, rlabels, n_pers, n_im):
    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(labels.reshape((n_pers, n_im)))
    plt.title("Ground truth")

    plt.subplot(132)
    plt.imshow(Y_masked.reshape((n_pers, n_im)))
    plt.title("Masked labels")

    plt.subplot(133)
    plt.imshow(rlabels.reshape((n_pers, n_im)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()


def offline_face_recognition(data='code_material_python/data/10faces/', \
                             classifier='code_material_python/data/haarcascade_frontalface_default.xml', extended=False, \
                             n_pers=2, n_im=2, n_l=4, \
                             var=1e5, gamma=.95, k=0, eps=0, lap="rw", \
                             frame_size=96, brightness=200, filter_method="Gaussian",  \
                             plot_the_dataset=False,  viz=True):
    """ Face recognition from SSL """

    faces, labels = _fetch_dataset(data, classifier, frame_size, brightness, extended, n_pers, n_im, filter_method)

    Y_masked = _random_masked_labels(n_pers, n_im, 4)
    rlabels = hard_hfs(faces, Y_masked, gamma, var=var, eps=eps, k=k, \
                      lap=lap)

    if plot_the_dataset:
        _plot_dataset(faces, labels, n_pers, n_im)
    if viz:
        _visualization_face_detection(labels, Y_masked, rlabels, n_pers, n_im)

    return np.equal(rlabels, labels).mean()
