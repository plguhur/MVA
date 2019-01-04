import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2 as cv
import os
import sys
from scipy.spatial import distance
import scipy.io as sio

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *

face_haar_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_haar_cascade = cv.CascadeClassifier("data/haarcascade_eye.xml")



def create_user_profile(user_name, faces_path = "data/"):
    ## Find the "faces" directory
    assert ("faces" in os.listdir(faces_path)), "Error : 'faces' folder not found"
    ## Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    image_count = 0
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
        print("New profile created at path", profile_path)
    else:
        image_count = len(os.listdir(profile_path))
        print("Profile found with", image_count, "images.")
    ## Launch video capture
    cam = cv.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        working_image = cv.bilateralFilter(grey_image, 9, 75, 75)
        working_image = cv.equalizeHist(working_image)
        working_image = cv.GaussianBlur(working_image, (5, 5), 0)
        box = face_haar_cascade.detectMultiScale(working_image)
        if len(box) > 0:
            box_surface = box[:, 2] * box[:, 3]
            index = box_surface.argmax()
            b0 = box[index]
            cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]), (0, 255, 0), 2)
        cv.putText(img,"[s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]: break  # esc or e to quit
        if key == ord('s'):
            ## Save face
            if len(box) > 0:
                x, y = b0[0], b0[1]
                x_range, y_range = b0[2], b0[3]
                image_count = image_count + 1
                image_name = os.path.join(profile_path, "img_" + str(image_count) + ".bmp")
                img_to_save = img[y:(y+y_range), x:(x+x_range)]
                cv.imwrite(image_name, img_to_save)
                print("Image", image_count, "saved at", image_name)
    cv.destroyAllWindows()
    return

def load_profile(user_name, faces_path = "data/"):
    assert ("faces" in os.listdir(faces_path)), "Error : 'faces' folder not found"
    ## Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    if not os.path.exists(profile_path):
        raise Exception("Profile not found")
    image_count = len(os.listdir(profile_path))
    print("Profile found with", image_count, "images.")
    images = [ os.path.join(profile_path, x) for x in os.listdir(profile_path) ]
    rep = np.zeros( (len(images), 96*96) )
    for i, im_path in enumerate(images):
        im = cv.imread(im_path, 0)
        cv.waitKey(1)
        rep[i, :] = preprocess_face(im)
    return rep


class incremental_k_centers:
    def __init__(self, labeled_faces, labels, max_num_centroids = 50):
        ## Number of labels to cluster
        self.n_labels = max(labels)
        ## Dimension of the input image
        self.image_dimension = labeled_faces.shape[1]
        ## Check input validity
        assert (set(labels) == set(range(1, 1 + self.n_labels))), "Initially provided faces should be labeled in [1, max]"
        assert (len(labeled_faces) == len(labels)), "Initial faces and initial labels are not of same size"
        ## Number of labelled faces

        self.n_labeled_faces = len(labeled_faces)
        ## Model parameter : number of maximum stored centroids
        self.max_num_centroids = max_num_centroids
        ## Model centroids (inital labeled faces)
        self.centroids = labeled_faces
        ## Centroids labels
        self.Y = labels
        ## Compute all the distances
        self.centroids_distances = None
        self.init = True
        print('[s]ave a frame ?')
        #self.taboo = (np.zeros(self.n_labeled_faces) == 0)

    def online_ssl_update_centroids(self, face):
    #
    # Input

    # self:
    #     the current cover state
    # face:
    #     the new sample
    # Output
    # No output, update self

        assert (self.image_dimension == len(face)), "new image not of good size"

        if len(self.centroids) >= self.max_num_centroids + 1:

            # Initialization
            if self.init:
                ## Compute the centroids distances
                self.centroids_distances = distance.cdist(self.centroids, self.centroids)
                ## set labeled nodes and self loops as infinitely distant
                np.fill_diagonal(self.centroids_distances, +np.Inf)
                self.centroids_distances[0:self.n_labeled_faces, 0:self.n_labeled_faces] = +np.Inf
                ## put labeled nodes in the taboo list
                self.taboo = np.array(range(self.centroids.shape[0])) < self.n_labeled_faces
                ## initialize multiplicity
                self.V = np.ones(self.centroids.shape[0])
                self.init = False
            # find the position of the two closest vertices
            #################################################################
            # find the edge (c_rep,c_add) with the minimum distance
            # (maximum similarity)                                          #
            #################################################################
            arg_min = np.argmin(self.centroids_distances)
            c_rep, c_add = np.unravel_index(arg_min, \
                                self.centroids_distances.shape)
            # min_dist = self.centroids_distances[c_rep, c_add]

            #################################################################
            #################################################################

            #################################################################
            # update data structures                                        #
            # None of the two nodes are taboo, if there is one, it is c_rep,
            # otherwise, c_rep is the bigest centroid
            #################################################################
            if c_add < self.n_labeled_faces and c_rep < self.n_labeled_faces:
                raise ValueError

            if c_add < self.n_labeled_faces or self.V[c_rep] < self.V[c_add]:
                c_add, c_rep = c_rep, c_add

            #################################################################
            #################################################################

            #################################################################
            # c_rep absorbe c_add, c_add is now the new face update
            # centroids and V
            #################################################################
            self.V[c_rep] += self.V[c_add]
            self.V[c_add] = 1
            self.centroids[c_add] = face

            #################################################################
            #################################################################
            ## Update the matrix distance
            dist_row = distance.cdist([self.centroids[c_add]], self.centroids)[0]
            dist_row[c_add] = +np.inf
            self.centroids_distances[c_add, :] = dist_row
            self.centroids_distances[:, c_add] = dist_row
            self.last_face = c_add

        else:
            ## Just add the new face as a centroid
            current_len = len(self.centroids)
            self.Y = np.append(self.Y, 0)
            self.centroids = np.vstack( [self.centroids, face] )
            # self.taboo = self.centroids

    def online_ssl_compute_solution(self):
    # Output
    # A label vector prediction of self.last_face
        ###################################
        # choose the experiment parameter
        ###################################
        eps = 0
        var = 1e5
        k = 10
        ###################################
        ###################################
        W = build_similarity_graph(self.centroids, var=var, eps=eps, k=k)
        if self.init:
            V = np.diag(np.ones(self.centroids.shape[0]))
            self.last_face = self.centroids.shape[0] - 1
        else:
            V = np.diag(self.V)
        W = V.dot(W.dot(V))
        ###################################
        # build the laplacian, compute the
        # hardHFS solution f (helper contain
        # an hardHFS function)
        ###################################
        L = build_laplacian(W, laplacian_normalization="rw")
        f = hardHFS(self.centroids, self.Y, L)
        ###################################
        ###################################
        return f[self.last_face]



def online_face_recognition(profile_names, n_pictures = 10):
    images = []
    labels = []
    label_names = []
    for i, name in enumerate(profile_names):
        p = load_profile(name)
        p = p[0:n_pictures, ]
        images += [p]
        labels += [ np.ones(p.shape[0]) * (i + 1) ]
        label_names += [name]
    faces = np.vstack(images)
    labels = np.hstack(labels).astype(np.int)
    ## Generate model
    model = incremental_k_centers(faces, labels)
    ## Start camera
    cam = cv.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        working_image = cv.bilateralFilter(grey_image, 9, 75, 75)
        working_image = cv.equalizeHist(working_image)
        working_image = cv.GaussianBlur(working_image, (5, 5), 0)
        box = face_haar_cascade.detectMultiScale(working_image)
        for b0 in box:
            x, y = b0[0], b0[1]
            x_range, y_range = b0[2], b0[3]
            ## look for eye classifier
            local_image = img[y:(y+y_range), x:(x+x_range)]
            eye_box = eye_haar_cascade.detectMultiScale(local_image)
            if len(eye_box) == 0:
                cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]), (0, 0, 255), 2)
                continue
            ## select face
            local_image = grey_image[y:(y+y_range), x:(x+x_range)]
            x_t = preprocess_face(local_image)
            model.online_ssl_update_centroids(x_t)
            p1, p2 = tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4])
            ###################################
            f=model.online_ssl_compute_solution()
            lab=np.argsort(f)
            ####################################################################################
            # Change False by something else to be able to disregard faces it cannot recognize #
            ####################################################################################
            if f[lab[0]] < 0.5:
            ####################################################################################
            ####################################################################################
               color = (100, 100, 100)
               txt = "unknown"
               cv.putText(img, txt , (p1[0] , p1[1] -5 ), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color)
            else:
               for i,l in enumerate(lab):
                color = [(0, 255, 0),(255, 0,0),(0, 0, 255)][l]
                txt = label_names[l] + "  " + ('%.4f' % np.abs(f[l]))
                cv.putText(img, txt , (p1[0] , p1[1] -5 - 10*i), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5+0.5*(i==f.shape[0]-1), color)
            cv.rectangle(img, p1, p2, color, 2)
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]: break
        if key == ord('s'):
            ## Save face
                print('saved')
                cv.imwrite("frame.png", img)
                ## cv.waitKey(1)
    cv.destroyAllWindows()

def preprocess_face(grey_face):
#     Transforms a n x n image into a feature vector
#
# Input
# gray_face:
#     ( n x n ) image in grayscale
#
# Output
# gray_face_vector:
#     ( 1 x EXTR_FRAME_SIZE^2) row vector with the preprocessed face
# Face preprocessing
    EXTR_FRAME_SIZE = 96
#######################################################################
# Apply preprocessing to balance the image (color/lightning), such    #
# as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and #
# equalization (cv.equalizeHist).                                     #
#######################################################################
    grey_face = cv.resize(grey_face, (EXTR_FRAME_SIZE, EXTR_FRAME_SIZE), \
                    interpolation = cv.INTER_CUBIC)
    grey_face = cv.boxFilter(grey_face, -1, (5, 5))
    grey_face = cv.GaussianBlur(grey_face, (5, 5), 0)
    grey_face = cv.equalizeHist(grey_face)
#######################################################################
#######################################################################
# resize the face
    grey_face = grey_face.reshape(EXTR_FRAME_SIZE * EXTR_FRAME_SIZE)
# scale the data to [0,1]
    grey_face = grey_face / 256
    return grey_face
