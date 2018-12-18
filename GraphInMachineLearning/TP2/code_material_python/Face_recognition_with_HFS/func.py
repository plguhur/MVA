import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2
import os
import sys

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *
from harmonic_function_solution.func import *

def offline_face_recognition():
#     a skeleton function to test offline face recognition, needs to be completed

    # Parameters
    cc = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
    frame_size = 96
    gamma = .95
    # Loading images
    images = np.zeros((100, frame_size ** 2))
    labels = np.zeros(100)
    var=10000

    for i in np.arange(10):
      for j in np.arange(10):
        im = sm.imread("../data/10faces/%d/%02d.jpg" % (i, j + 1))
        box = cc.detectMultiScale(im)
        top_face = {"area": 0}
    
    
    
        for cfx, cfy, clx, cly in box:
            face_area = clx * cly
            if face_area > top_face["area"]:
                top_face["area"] = face_area
                top_face["box"] = [cfx, cfy, clx, cly]

        fx, fy, lx, ly = top_face["box"]
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_face = gray_im[fy:fy + ly, fx:fx + lx]
        
        #######################################################################
        # Apply preprocessing to balance the image (color/lightning), such    #
        # as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and #
        # equalization (cv.equalizeHist).                                     #
        #######################################################################

        #######################################################################
        #######################################################################

        #resize the face and reshape it to a row vector, record labels
        images[j * 10 + i] = gf.reshape((-1))
        labels[j * 10 + i] = i + 1
        
        
        
    # if you want to plot the dataset, set the following variable to 1
    ##################################################################
    plot_the_dataset = 
    ##################################################################
    ##################################################################
    if plot_the_dataset:

     pyplot.figure(1)
     for i in range(100):
        pyplot.subplot(10,10,i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i].reshape(frame_size,frame_size))
        r='{:d}'.format(i+1)
        if i<10:
         pyplot.title('Person '+r)
     pyplot.show()  
 
 
        

    #################################################################
    # select 4 random labels per person and reveal them             #
    # Y_masked: (n x 1) masked label vector, where entries Y_i      #
    #       takes a value in [1..num_classes] if the node is        #
    #       labeled, or 0 if the node is unlabeled (masked)         #
    #################################################################

    #######################################################################
    #######################################################################
    #################################################################
    # choose the experiment parameter and                           #
    # compute hfs solution using either soft_hfs or hard_hfs        #
    #################################################################




    # Plots #
    pyplot.subplot(121)
    pyplot.imshow(labels.reshape((10, 10)))

    pyplot.subplot(122)
    pyplot.imshow(rlabels.reshape((10, 10)))
    pyplot.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    pyplot.show()
    
    
    
    
    
    
    
    
def offline_face_recognition_augmented():
#     a skeleton function to test offline face recognition, needs to be completed

    # Parameters
    cc = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
    frame_size = 96
    gamma = .95
    nbimgs = 50
    # Loading images
    images = np.zeros((10 * nbimgs, frame_size ** 2))
    labels = np.zeros(10 * nbimgs)
    var=10000

    for i in np.arange(10):
      imgdir = "../data/extended_dataset/%d" % i
      imgfns = os.listdir(imgdir)
      for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
        im = sm.imread("{}/{}".format(imgdir, imgfn))
        box = cc.detectMultiScale(im)
        top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

        for cfx, cfy, clx, cly in box:
            face_area = clx * cly
            if face_area > top_face["area"]:
                top_face["area"] = face_area
                top_face["box"] = [cfx, cfy, clx, cly]


        fx, fy, lx, ly = top_face["box"]
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_face = gray_im[fy:fy + ly, fx:fx + lx]
        
        #######################################################################
        # Apply preprocessing to balance the image (color/lightning), such    #
        # as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and #
        # equalization (cv.equalizeHist).                                     #
        #######################################################################

        #######################################################################
        #######################################################################

        #resize the face and reshape it to a row vector, record labels
        images[j * 10 + i] = gf.reshape((-1))
        labels[j * 10 + i] = i + 1
        
        
        
    # if you want to plot the dataset, set the following variable to 1

    plot_the_dataset = 0

    if plot_the_dataset:

     pyplot.figure(1)
     for i in range(10 * nbimgs):
        pyplot.subplot(nbimgs,10,i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i].reshape(frame_size,frame_size))
        r='{:d}'.format(i+1)
        if i<10:
         pyplot.title('Person '+r)
     pyplot.show()  
 
 
        

    #################################################################
    # select 4 random labels per person and reveal them             #
    # Y_masked: (n x 1) masked label vector, where entries Y_i      #
    #       takes a value in [1..num_classes] if the node is        #
    #       labeled, or 0 if the node is unlabeled (masked)         #
    #################################################################

    #######################################################################
    #######################################################################
    #################################################################
    # choose the experiment parameter and                           #
    # compute hfs solution using either soft_hfs or hard_hfs        #
    #################################################################




    # Plots #
    pyplot.subplot(121)
    pyplot.imshow(labels.reshape((-1, 10)))

    pyplot.subplot(122)
    pyplot.imshow(rlabels.reshape((-1, 10)))
    pyplot.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    pyplot.show()
