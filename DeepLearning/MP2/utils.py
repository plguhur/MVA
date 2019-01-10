import matplotlib.pyplot as plt
import numpy as np

def decode_with_labels(path, split=True):
    with open(path, 'r') as f:
        lines = f.readlines()
    label = [int(line[0]) for line in lines]
    words = [line[2:].split() for line in lines] if split \
                else [line[2:] for line in lines]
    return words, label

def decode_without_labels(path, split=True):
    with open(path, 'r') as f:
        lines = f.readlines()
    if split:
        return [line[2:].split() for line in lines]
    else:
        return [line[2:] for line in lines]


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
