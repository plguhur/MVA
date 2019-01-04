# This code comes from https://github.com/akmtn/pytorch-siggraph2017-inpainting/blob/master/inpaint.py


import argparse
import os
import torch
from torch.legacy import nn
from torch.legacy.nn.Sequential import Sequential
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import torchvision.utils as vutils
try:
    from poissonblending import prepare_mask, blend
except ModuleNotFoundError:
    from src.poissonblending import prepare_mask, blend

def tensor2cvimg(src):
    '''return np.array
        uint8
        [0, 255]
        BGR
        (H, W, C)
    '''
    out = src.copy() * 255
    out = out.transpose((1, 2, 0)).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return out

def cvimg2tensor(src):
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = out.transpose((2,0,1)).astype(np.float64)
    out = out / 255

    return out

# load Completion Network
def load_network(model_path="completionnet_places2.t7"):
    data = load_lua(model_path)
    model = data.model
    model.evaluate()
    datamean = data.mean
    return model, datamean

def random_mask(output_shape):
    # generate random holes
    w, h = output_shape
    M = torch.FloatTensor(1, h, w).fill_(0)
    nHoles = np.random.randint(1, 4)
    print("Random mask with", nHoles, "holes")
    for _ in range(nHoles):
        mask_w = np.random.randint(32, 128)
        mask_h = np.random.randint(32, 128)
        assert h > mask_h or w > mask_w
        px = np.random.randint(0, w-mask_w)
        py = np.random.randint(0, h-mask_h)
        M[:, py:py+mask_h, px:px+mask_w] = 1
    return M

# load data
def load_data(input_path, output_shape=None):
    input_img = cv2.imread(input_path)
    if output_shape is not None:
        input_img = cv2.resize(input_img, output_shape)
    I = torch.from_numpy(cvimg2tensor(input_img)).float()
    return I


def post_processing(I, M, out):
    M_3ch = torch.cat((M, M, M), 0)
    target = tensor2cvimg(I.numpy())
    source = tensor2cvimg(out.numpy())    # foreground
    mask = tensor2cvimg(M_3ch.numpy())
    out = blend(target, source, mask, offset=(0, 0))
    return torch.from_numpy(cvimg2tensor(out))

def load_mask(mask_path, output_shape=None):
    M =  cv2.imread(mask_path)
    if output_shape is not None:
        M = cv2.resize(M, output_shape)
    M = torch.from_numpy(
                cv2.cvtColor(M, cv2.COLOR_BGR2GRAY) / 255).float()
    M[M <= 0.2] = 0.0
    M[M > 0.2] = 1.0
    M = M.view(1, M.size(0), M.size(1))
    return M

def inpainting(model, datamean, I, M, gpu=False, postproc=False):

    assert I.size(1) == M.size(1) and I.size(2) == M.size(2)

    for i in range(3):
        I[i, :, :] = I[i, :, :] - datamean[i]

    # make mask_3ch
    M_3ch = torch.cat((M, M, M), 0)

    Im = I * (M_3ch*(-1)+1)
    # set up input
    input = torch.cat((Im, M), 0)
    input = input.view(1, input.size(0), input.size(1), input.size(2)).float()

    if gpu:
        print('using GPU...')
        model.cuda()
        input = input.cuda()

    # evaluate
    res = model.forward(input)[0].cpu()

    # make out
    for i in range(3):
        I[i, :, :] = I[i, :, :] + datamean[i]

    out = res.float()*M_3ch.float() + I.float()*(M_3ch*(-1)+1).float()

    # post-processing
    if postproc:
        print('post-postprocessing...')
        out = post_processing(I, M, out)

    return out


if __name__ == "__main__":

    model, datamean = load_network()
    I = load_data("images/bridge.jpg", output_shape=(600, 400))
    # I = load_data("images/example.png")
    M = random_mask(I.shape[1:])
    out = inpainting(model, datamean, I, M)

    print('save images...')
    vutils.save_image(out, 'out.png', normalize=True)
    # vutils.save_image(Im, 'masked_input.png', normalize=True)
    vutils.save_image(M, 'mask.png', normalize=True)
    # vutils.save_image(res, 'res.png', normalize=True)
    print('Done')
