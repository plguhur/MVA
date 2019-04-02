# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
#
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

from scipy import ndimage
import numpy as np
import cv2
import os
from PIL import Image


def random_tps(image):
    w, h = image.size
    image = np.array(image)
    def_img = warp_images([image[:, :, 0], image[:, :, 1], image[:, :, 2]],
                          (0, 0, h - 1, w - 1), interpolation_order=1, approximate_grid=2)
    img = np.transpose(def_img, axes=(1, 2, 0)).copy()
    img = Image.fromarray(img.astype(np.uint8))
    return img



def deform_grid(h, w, n=3):
    """ creates regular grid, applies random perturbation on it """
    fact = 0.05
    bound = min(w, h) * fact
    vec1 = (h / (n - 1)) * np.arange(n)
    vec2 = (w / (n - 1)) * np.arange(n)
    grid = np.transpose([np.repeat(vec1, n), np.tile(vec2, n)])
    new_grid = np.zeros_like(grid)
    for i in range(n * n):
        y = grid[i, 0]
        x = grid[i, 1]
        new_grid[i] = grid[i]
        if 0. < x < w:
            new_grid[i, 1] += np.random.uniform(-bound, bound)
        if 0. < y < h:
            new_grid[i, 0] += np.random.uniform(-bound, bound)
        # print('{} ----> {}'.format(grid[i], new_grid[i]))
    return grid, new_grid

def warp_images(images, output_region, interpolation_order=1, approximate_grid=2):
    """Define a thin-plate-spline warping transform that warps from the from_points
    to the to_points, and then warp the given images by that transform. This
    transform is described in the paper: "Principal Warps: Thin-Plate Splines and
    the Decomposition of Deformations" by F.L. Bookstein.
    Parameters:
        - from_points and to_points: Nx2 arrays containing N 2D landmark points.
        - images: list of images to warp with the given warp transform.
        - output_region: the (xmin, ymin, xmax, ymax) region of the output
                image that should be produced. (Note: The region is inclusive, i.e.
                xmin <= x <= xmax)
        - interpolation_order: if 1, then use linear interpolation; if 0 then use
                nearest-neighbor.
        - approximate_grid: defining the warping transform is slow. If approximate_grid
                is greater than 1, then the transform is defined on a grid 'approximate_grid'
                times smaller than the output image region, and then the transform is
                bilinearly interpolated to the larger region. This is fairly accurate
                for values up to 10 or so.
    """
    
    h, w = images[0].shape
    from_points, to_points = deform_grid(h, w)
    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    return [ndimage.map_coordinates(np.asarray(image), transform, order=interpolation_order) for image in images]


def indices(a, b):
    return np.transpose(np.asarray((a, b)), axes=(1, 2, 0))


def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    x_min, y_min, x_max, y_max = output_region
    if approximate_grid is None:
        approximate_grid = 1
    x_steps = (x_max - x_min) / approximate_grid
    y_steps = (y_max - y_min) / approximate_grid
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]

    # make the reverse transform warping from the to_points to the from_points, because we
    # do image interpolation in this reverse fashion
    transform = _make_warp(to_points, from_points, x, y)

    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y = np.mgrid[x_min:x_max+1, y_min:y_max+1]
        x_fracs, x_indices = np.modf((x_steps-1)*(new_x-x_min)/float(x_max-x_min))
        y_fracs, y_indices = np.modf((y_steps-1)*(new_y-y_min)/float(y_max-y_min))
        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        ix1 = (x_indices+1).clip(0, x_steps-1).astype(int)
        iy1 = (y_indices+1).clip(0, y_steps-1).astype(int)
        t00 = transform[0][(x_indices, y_indices)]
        t01 = transform[0][(x_indices, iy1)]
        t10 = transform[0][(ix1, y_indices)]
        t11 = transform[0][(ix1, iy1)]
        transform_x = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs
        t00 = transform[1][(x_indices, y_indices)]
        t01 = transform[1][(x_indices, iy1)]
        t10 = transform[1][(ix1, y_indices)]
        t11 = transform[1][(ix1, iy1)]
        transform_y = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs
        transform = [transform_x, transform_y]
    return transform


_small = 1e-100


def _U(x):
    return (x**2) * np.where(x < _small, 0, np.log(x))


def _interpoint_distances(points):
    xd = np.subtract.outer(points[:, 0], points[:, 0])
    yd = np.subtract.outer(points[:, 1], points[:, 1])
    return np.sqrt(xd**2 + yd**2)


def _make_L_matrix(points):
    n = len(points)
    k = _U(_interpoint_distances(points))
    p = np.ones((n, 3))
    p[:, 1:] = points
    o = np.zeros((3, 3))
    ll = np.asarray(np.bmat([[k, p], [p.transpose(), o]]))
    return ll


def _calculate_f(coeffs, points, x, y):
    w = coeffs[:-3]
    a1, ax, ay = coeffs[-3:]
    # The following uses too much RAM:
    # distances = _U(numpy.sqrt((points[:,0]-x[...,numpy.newaxis])**2 + (points[:,1]-y[...,numpy.newaxis])**2))
    # summation = (w * distances).sum(axis=-1)
    summation = np.zeros(x.shape)
    for wi, Pi in zip(w, points):
        summation += wi * _U(np.sqrt((x-Pi[0])**2 + (y-Pi[1])**2))
    return a1 + ax*x + ay*y + summation


def _make_warp(from_points, to_points, x_vals, y_vals):
    from_points, to_points = np.asarray(from_points), np.asarray(to_points)
    err = np.seterr(divide='ignore')
    ll = _make_L_matrix(from_points)
    v = np.resize(to_points, (len(to_points)+3, 2))
    v[-3:, :] = 0
    coeffs = np.dot(np.linalg.pinv(ll), v)
    x_warp = _calculate_f(coeffs[:, 0], from_points, x_vals, y_vals)
    y_warp = _calculate_f(coeffs[:, 1], from_points, x_vals, y_vals)
    np.seterr(**err)
    return [x_warp, y_warp]


def deform(img):
    n = 5
    h, w = img.shape[:2]
    fact = 0.05
    bound = min(w, h) * fact

    vec1 = (h / (n - 1)) * np.arange(n)
    vec2 = (w / (n - 1)) * np.arange(n)
    grid = np.transpose([np.repeat(vec1, n), np.tile(vec2, n)])
    new_grid = np.zeros_like(grid)
    for i in range(n * n):
        y = grid[i, 0]
        x = grid[i, 1]
        new_grid[i] = grid[i]
        if 0. < x < w:
            new_grid[i, 1] += np.random.uniform(-bound, bound)
        if 0. < y < h:
            new_grid[i, 0] += np.random.uniform(-bound, bound)
        # print('{} ----> {}'.format(grid[i], new_grid[i]))
    res = warp_images(grid, new_grid,
                      [img[:, :, 0], img[:, :, 1], img[:, :, 2]],
                      (0, 0, h, w), interpolation_order=1, approximate_grid=2)
    res_img = np.transpose(res, axes=(1, 2, 0)).copy()
    # for i in range(n*n):
    #     y, x = grid[i].astype(np.int)
    #     ny, nx = new_grid[i].astype(np.int)
    #     cv2.arrowedLine(res_img, (x, y), (nx, ny), (0, 0, 255), 3)
    return res_img