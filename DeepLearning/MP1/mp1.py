
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm, tqdm_notebook
from keras.utils import np_utils

# On some implementations of matplotlib, you may need to change this value
IMAGE_SIZE = 72


def plot_history(history):
    if 'acc' in history.history:
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)

        plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()



def generate_a_drawing(figsize, U, V, noise=0.0, reshape=False):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    if reshape:
        imdata = imdata.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False):
    figsize = 1.0
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(figsize, U, V, noise)


def generate_a_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise)
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]

def generate_triangles(n_samples, noise=0.0, free_location=False):
    figsize = 1.0

    if free_location:
        U = np.random.random((3, n_samples))
        V = np.random.random((3, n_samples))
    else:
        size = (0.3 + 0.7 * np.random.random(n_samples))*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    return U, V


def generate_dataset_classification(n_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([n_samples,im_size])
    Y = np.zeros(n_samples)

    for i in tqdm(range(n_samples), "Creating data: "):
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1:
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    Y = np_utils.to_categorical(Y, 3)
    return [X, Y]

def generate_test_set_classification(n_samples, noise=0.0, free_location=False):
    np.random.seed(42)
    return generate_dataset_classification(n_samples, noise, free_location)

def generate_dataset_regression(n_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([n_samples,im_size])
    Y = np.zeros([n_samples, 6])

    for i in tqdm(range(n_samples), "Creating data: "):
        [X[i], Y[i]] = generate_a_triangle(noise, free_location)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]


def visualize_prediction(x, y, show=True, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((IMAGE_SIZE,IMAGE_SIZE))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)

    plt.show()

def visualize_predictions(X, Y, ground_truth=None):
    fig, ax = plt.subplots(figsize=(20, 10))
    n_samples = len(X)
    for i in range(n_samples):
        ax = plt.subplot(1,n_samples, i+1)
        plt.axis('off')
        I = X[i].reshape((IMAGE_SIZE,IMAGE_SIZE))
        ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        xy = Y[i].reshape(3,2)
        tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
        ax.add_patch(tri)

        if ground_truth is not None:
            xy = ground_truth[i].reshape(3,2)
            tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'g', linewidth = 5, alpha = 0.5)
            ax.add_patch(tri)

    plt.show()




def visualize_denoising(X, Y, ground_truth=None):
    fig, ax = plt.subplots(figsize=(20, 10))
    n_samples = len(X)
    n_col = 2 if ground_truth is None else 3
    for i in range(n_samples):
        ax = plt.subplot(n_samples, n_col, n_col*i+1)
        I = X[i].reshape((IMAGE_SIZE,IMAGE_SIZE))
        ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
        I = Y[i].reshape((IMAGE_SIZE,IMAGE_SIZE))
        ax = plt.subplot(n_samples, n_col, n_col*i+2)
        ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
        if ground_truth is not None:
            ax = plt.subplot(n_samples, n_col, n_col*i+3)
            I = ground_truth[i].reshape((IMAGE_SIZE,IMAGE_SIZE))
            ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    plt.show()


def visualize_pred_gt(x, y, y_pred):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((IMAGE_SIZE,IMAGE_SIZE))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'g', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)
    xy = y_pred.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)

    plt.show()


def generate_test_set_regression(n_samples, noise=0.0, free_location=False):
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(n_samples, noise, free_location)
    return [X_test, Y_test]


# def generate_dataset_noise(n_samples, noise=20.0, free_location=False):
#     np.random.seed(42)
#     U, V = generate_triangles(n_samples, noise=0.0, free_location=free_location)
#     U = np.swapaxes(U, 0, 1)
#     V = np.swapaxes(V, 0, 1)
#     X = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 1))
#     Y = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 1))
#     for i in tqdm(range(n_samples)):
#         Y[i] = generate_a_drawing(1.0, U[i], V[i], noise=0.0, reshape=True)
#         img = generate_a_drawing(1.0, U[i], V[i], noise=noise, reshape=True)
#         X[i] = (img + noise) / (255 + 2 * noise)
#     return X, Y

def generate_dataset_noise(n_samples, noise=20.0):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    Y = np.zeros([n_samples,im_size])

    for i in tqdm_notebook(range(n_samples)):
        [Y[i], _] = generate_a_triangle(0.0, True)
    noise /= 255.
    Y /= 255.
    X = Y + noise * (2 * np.random.random(im_size) - 1)
    X = (X + noise) / (1. + 2 * noise)
    return [X, Y]
