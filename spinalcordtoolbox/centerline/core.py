#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with centerline extraction from 3D data.


import sys

import numpy as np
from spinalcordtoolbox.image import Image, zeros_like
import sct_utils as sct


class ParamCenterline:
    def __init__(self, contrast=None, degree=3):
        self.contrast = contrast
        self.degree = degree  # Degree of polynomial function


def centermass_slicewise(x, y, z):
    """
    # Get the center of mass at each z. Note: x, y and z must have same length.
    :param x: 1d-array: X-coordinates
    :param y: 1d-array: Y-coordinates
    :param z: 1d-array: Z-coordinates
    :return: x_mean, y_mean, z_mean: 1d-array, 1d-array, 1d-array
    """
    assert len(x) == len(y) == len(z)
    x_mean, y_mean, z_mean = np.array([]), np.array([]), np.array([])
    # Loop across unique x values (and sort it)
    for iz in sorted(set(z)):
        # Get indices corresponding to iz
        ind_z = np.where(z == iz)
        if len(ind_z[0]):
            # Average all x and y values at ind_z
            x_mean = np.append(x_mean, x[ind_z].mean())
            y_mean = np.append(y_mean, y[ind_z].mean())
            z_mean = np.append(z_mean, iz)
    return x_mean, y_mean, z_mean


def get_centerline(segmentation, algo_fitting='polyfit', param=ParamCenterline(), verbose=1):
    """
    Extract centerline from an image (using optic) or from a binary or weighted segmentation (using the center of mass).
    :param segmentation: input segmentation or series of points along the centerline. Could be an Image or a file name.
    :param algo_fitting: str:
        polyfit: Polynomial fitting
        nurbs:
        optic: Automatic segmentation using SVM and HOG. See [Gros et al. MIA 2018].
    :param param: ParamCenterline()
    :param verbose: int: verbose level
    :return: im_centerline: Image: Centerline in discrete coordinate (int)
    :return: arr_centerline: 3x1 array: Centerline in continuous coordinate (float) for each slice
    :return: arr_centerline_deriv: 2x1 array: Derivatives of x and y centerline wrt. z for each slice
    """
    # Open image and change to RPI orientation
    im_seg = Image(segmentation)
    native_orientation = im_seg.orientation
    im_seg.change_orientation('RPI')
    px, py, pz = im_seg.dim[4:7]
    x, y, z = np.where(im_seg.data)
    z_ref = np.array(range(im_seg.dim[2]))

    # Take the center of mass at each slice to avoid: https://stackoverflow.com/questions/2009379/interpolate-question
    x_mean, y_mean, z_mean = centermass_slicewise(x, y, z)

    # Choose method
    if algo_fitting == 'polyfit':
        from spinalcordtoolbox.centerline.curve_fitting import polyfit_1d
        x_centerline_fit, x_centerline_deriv = polyfit_1d(z_mean, x_mean, z_ref, deg=param.degree)
        y_centerline_fit, y_centerline_deriv = polyfit_1d(z_mean, y_mean, z_ref, deg=param.degree)

    # elif algo_fitting == 'sinc':
    #     from spinalcordtoolbox.centerline.curve_fitting import sinc_interp
    #     z_ref = np.array(range(im_seg.dim[2]))
    #     x_centerline_fit, x_centerline_deriv = sinc_interp(z_mean, x_mean, z_ref)
    #     y_centerline_fit, y_centerline_deriv = sinc_interp(z_mean, y_mean, z_ref)

    elif algo_fitting == 'bspline':
        from spinalcordtoolbox.centerline.curve_fitting import bspline
        x_centerline_fit, x_centerline_deriv = bspline(z_mean, x_mean, z_ref, deg=param.degree)
        y_centerline_fit, y_centerline_deriv = bspline(z_mean, y_mean, z_ref, deg=param.degree)

    elif algo_fitting == 'nurbs':
        from spinalcordtoolbox.centerline.nurbs import b_spline_nurbs
        # TODO: do something about nbControl: previous default was -1.
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, \
            z_centerline_deriv, error = b_spline_nurbs(x_mean, y_mean, z_mean, nbControl=None)

    elif algo_fitting == 'optic':
        # This method is particular compared to the previous ones, as here we estimate the centerline based on the
        # image itself (not the segmentation). Hence, we can bypass the fitting procedure and centerline creation
        # and directly output results.
        from spinalcordtoolbox.centerline import optic
        from spinalcordtoolbox.centerline.curve_fitting import polyfit_1d
        im_centerline = optic.detect_centerline(im_seg, param.contrast)
        x_centerline_fit, y_centerline_fit, z_centerline = np.where(im_centerline.data)
        # Compute derivatives using polynomial fit
        _, x_centerline_deriv = polyfit_1d(z_ref, x_centerline_fit, z_ref, deg=5)
        _, y_centerline_deriv = polyfit_1d(z_ref, y_centerline_fit, z_ref, deg=5)
        return im_centerline, \
               np.array([x_centerline_fit, y_centerline_fit, z_ref]), \
               np.array([x_centerline_deriv, y_centerline_deriv]),

    # Display fig of fitted curves
    if verbose == 2:
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Algo=%s, Deg=%s" % (algo_fitting, param.degree))
        plt.plot(z_mean * pz, x_mean * px, 'ro')
        plt.plot(z_ref * pz, x_centerline_fit * px)
        plt.ylabel("X [mm]")
        plt.subplot(2, 1, 2)
        plt.plot(z_mean, y_mean, 'ro')
        plt.plot(z_ref, y_centerline_fit)
        plt.xlabel("Z [mm]")
        plt.ylabel("Y [mm]")
        plt.savefig('fig_centerline_' + datetime.now().strftime("%y%m%d%H%M%S%f") + '_' + algo_fitting + '.png')
        plt.close()

    # Create an image with the centerline
    im_centerline = im_seg.copy()
    im_centerline.data = np.zeros(im_centerline.data.shape)
    # assign value=1 to centerline
    im_centerline.data[round_and_clip(x_centerline_fit), round_and_clip(y_centerline_fit), z_ref] = 1
    # reorient centerline to native orientation
    im_centerline.change_orientation(native_orientation)
    # TODO: reorient output array in native orientation
    return im_centerline, \
           np.array([x_centerline_fit, y_centerline_fit, z_ref]), \
           np.array([x_centerline_deriv, y_centerline_deriv]),


def round_and_clip(arr):
    """
    Round to closest int, convert to dtype=int and clip to min/max values allowed by the list length
    :param arr:
    :return:
    """
    return np.clip(arr.round().astype(int), 0, len(arr)-1)


def _call_viewer_centerline(fname_in, interslice_gap=20.0):
    # TODO: input should be im, not fname
    from spinalcordtoolbox.gui.base import AnatomicalParams
    from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

    im_data = Image(fname_in)

    # Get the number of slice along the (IS) axis
    im_tmp = im_data.copy().change_orientation('RPI')
    _, _, nz, _, _, _, pz, _ = im_tmp.dim
    del im_tmp

    params = AnatomicalParams()
    # setting maximum number of points to a reasonable value
    params.num_points = np.ceil(nz * pz / interslice_gap) + 2
    params.interval_in_mm = interslice_gap
    params.starting_slice = 'top'

    im_mask_viewer = zeros_like(im_data)
    controller = launch_centerline_dialog(im_data, im_mask_viewer, params)
    fname_labels_viewer = sct.add_suffix(fname_in, '_viewer')

    if not controller.saved:
        sct.log.error('The viewer has been closed before entering all manual points. Please try again.')
        sys.exit(1)
    # save labels
    controller.as_niftii(fname_labels_viewer)

    return fname_labels_viewer