#!/usr/bin/env python
########################################################################################################################
#
# This file contains useful functions for shape analysis based on spinal cord segmentation.
# The main input of these functions is a small image containing the binary spinal cord segmentation,
# ideally centered in the image.
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-12-20
#
# About the license: see the file LICENSE.TXT
########################################################################################################################


import numpy as np
import sct_utils as sct
import os
import time
import math
from random import randint

import tqdm
from skimage import measure, filters
import shutil
import matplotlib.pyplot as plt
from itertools import compress
from sct_image import Image, set_orientation
from msct_types import Centerline
from sct_straighten_spinalcord import smooth_centerline


def smoothing(image, sigma=1.0):
    return filters.gaussian(image, sigma=sigma)


def properties2d(image, resolution=None, verbose=1):
    label_img = measure.label(np.transpose(image))
    regions = measure.regionprops(label_img)
    areas = [r.area for r in regions]
    ix = np.argsort(areas)
    if len(regions) != 0:
        sc_region = regions[ix[-1]]
        try:
            ratio_minor_major = sc_region.minor_axis_length / sc_region.major_axis_length
        except ZeroDivisionError:
            ratio_minor_major = 0.0

        area = sc_region.area
        diameter = sc_region.equivalent_diameter
        major_l = sc_region.major_axis_length
        minor_l = sc_region.minor_axis_length
        if resolution is not None:
            area *= resolution[0] * resolution[1]
            # TODO: compute length depending on resolution. Here it assume the patch has the same X and Y resolution
            diameter *= resolution[0]
            major_l *= resolution[0]
            minor_l *= resolution[0]

            size_grid = 8.0 / resolution[0]  # assuming the maximum spinal cord radius is 8 mm
        else:
            size_grid = int(2.4 * sc_region.major_axis_length)

        """
        import matplotlib.pyplot as plt
        plt.imshow(label_img)
        plt.text(1, 1, sc_region.orientation, color='white')
        plt.show()
        """

        y0, x0 = sc_region.centroid
        orientation = sc_region.orientation

        resolution_grid = 0.25
        x_grid, y_grid = np.mgrid[-size_grid:size_grid:resolution_grid, -size_grid:size_grid:resolution_grid]
        coordinates_grid = np.array(list(zip(x_grid.ravel(), y_grid.ravel())))
        coordinates_grid_image = np.array([[x0 + math.cos(orientation) * coordinates_grid[i, 0], y0 - math.sin(orientation) * coordinates_grid[i, 1]] for i in range(coordinates_grid.shape[0])])

        from scipy.ndimage import map_coordinates
        square = map_coordinates(image, coordinates_grid_image.T, output=np.float32, order=0, mode='constant', cval=0.0)
        square_image = square.reshape((len(x_grid), len(x_grid)))

        size_half = square_image.shape[1] / 2
        left_image = square_image[:, :size_half]
        right_image = np.fliplr(square_image[:, size_half:])

        dice_symmetry = np.sum(left_image[right_image == 1]) * 2.0 / (np.sum(left_image) + np.sum(right_image))

        """
        import matplotlib.pyplot as plt
        plt.imshow(square_image)
        plt.text(3, 3, dice, color='white')
        plt.show()
        """

        sc_properties = {'area': area,
                         'bbox': sc_region.bbox,
                         'centroid': sc_region.centroid,
                         'eccentricity': sc_region.eccentricity,
                         'equivalent_diameter': diameter,
                         'euler_number': sc_region.euler_number,
                         'inertia_tensor': sc_region.inertia_tensor,
                         'inertia_tensor_eigvals': sc_region.inertia_tensor_eigvals,
                         'minor_axis_length': minor_l,
                         'major_axis_length': major_l,
                         'moments': sc_region.moments,
                         'moments_central': sc_region.moments_central,
                         'orientation': sc_region.orientation * 180.0 / math.pi,
                         'perimeter': sc_region.perimeter,
                         'ratio_minor_major': ratio_minor_major,
                         'solidity': sc_region.solidity,  # convexity measure
                         'symmetry': dice_symmetry
                         }
    else:
        sc_properties = None

    return sc_properties


def compute_properties_along_centerline(fname_seg_image, property_list, fname_discs='', smooth_factor=5.0, interpolation_mode=0, remove_temp_files=1, verbose=1):

    # Check list of properties
    # If diameters is in the list, compute major and minor axis length and check orientation
    compute_diameters = False
    property_list_local = list(property_list)
    if 'diameters' in property_list_local:
        compute_diameters = True
        property_list_local.remove('diameters')
        property_list_local.append('major_axis_length')
        property_list_local.append('minor_axis_length')
        property_list_local.append('orientation')

    # TODO: make sure fname_segmentation and fname_disks are in the same space
    path_tmp = sct.tmp_create(basename="compute_properties_along_centerline", verbose=verbose)

    sct.copy(fname_seg_image, path_tmp)
    if fname_discs:
        sct.copy(fname_discs, path_tmp)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    fname_segmentation = os.path.abspath(fname_seg_image)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', verbose)
    im_seg = Image(file_data + ext_data)
    fname_segmentation_orient = 'segmentation_rpi' + ext_data
    image = set_orientation(im_seg, 'RPI')
    image.setFileName(fname_segmentation_orient)
    image.save()

    # Initiating some variables
    nx, ny, nz, nt, px, py, pz, pt = image.dim
    resolution = 0.5
    properties = {key: [] for key in property_list_local}
    properties['incremental_length'] = []
    properties['distance_from_C1'] = []
    properties['vertebral_level'] = []
    properties['z_slice'] = []

    # compute the spinal cord centerline based on the spinal cord segmentation
    number_of_points = 5 * nz
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = \
        smooth_centerline(fname_segmentation_orient, algo_fitting='nurbs', verbose=verbose,
                          nurbs_pts_number=number_of_points, all_slices=False, phys_coordinates=True,
                          remove_outliers=True)
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

    # Compute vertebral distribution along centerline based on position of intervertebral disks
    if fname_discs:
        fname_disks = os.path.abspath(fname_discs)
        path_data, file_data, ext_data = sct.extract_fname(fname_disks)
        im_disks = Image(file_data + ext_data)
        fname_disks_orient = 'disks_rpi' + ext_data
        image_disks = set_orientation(im_disks, 'RPI')
        image_disks.setFileName(fname_disks_orient)
        image_disks.save()

        image_disks = Image(fname_disks_orient)
        coord = image_disks.getNonZeroCoordinates(sorting='z', reverse_coord=True)
        coord_physical = []
        for c in coord:
            c_p = image_disks.transfo_pix2phys([[c.x, c.y, c.z]])[0]
            c_p.append(c.value)
            coord_physical.append(c_p)
        centerline.compute_vertebral_distribution(coord_physical)

    sct.printv('Computing spinal cord shape along the spinal cord...')
    with tqdm.tqdm(total=centerline.number_of_points) as pbar:

        # Extracting patches perpendicular to the spinal cord and computing spinal cord shape
        for index in range(centerline.number_of_points):
            # value_out = -5.0
            value_out = 0.0
            current_patch = centerline.extract_perpendicular_square(image, index, resolution=resolution, interpolation_mode=interpolation_mode, border='constant', cval=value_out)

            # check for pixels close to the spinal cord segmentation that are out of the image
            from skimage.morphology import dilation
            patch_zero = np.copy(current_patch)
            patch_zero[patch_zero == value_out] = 0.0
            # patch_borders = dilation(patch_zero) - patch_zero

            """
            if np.count_nonzero(patch_borders + current_patch == value_out + 1.0) != 0:
                c = image.transfo_phys2pix([centerline.points[index]])[0]
                print('WARNING: no patch for slice', c[2])
                continue
            """

            sc_properties = properties2d(patch_zero, [resolution, resolution])
            if sc_properties is not None:
                properties['incremental_length'].append(centerline.incremental_length[index])
                if fname_discs:
                    properties['distance_from_C1'].append(centerline.dist_points[index])
                    properties['vertebral_level'].append(centerline.l_points[index])
                properties['z_slice'].append(image.transfo_phys2pix([centerline.points[index]])[0][2])
                for property_name in property_list_local:
                    properties[property_name].append(sc_properties[property_name])
            else:
                c = image.transfo_phys2pix([centerline.points[index]])[0]
                print('WARNING: no properties for slice', c[2])

            pbar.update(1)

    # Adding centerline to the properties for later use
    # UPDATE JULIEN: removed the line below because this property has different type than other properties (Centerline
    # vs. array), causing troubles later in the code. Would be better to export it in a different way.
    # properties['centerline'] = centerline

    # We assume that the major axis is in the right-left direction
    # this script checks the orientation of the spinal cord and invert axis if necessary to make sure the major axis is right-left
    if compute_diameters:
        diameter_major = properties['major_axis_length']
        diameter_minor = properties['minor_axis_length']
        orientation = properties['orientation']
        for i, orientation_item in enumerate(orientation):
            if -45.0 < orientation_item < 45.0:
                continue
            else:
                temp = diameter_minor[i]
                properties['minor_axis_length'][i] = diameter_major[i]
                properties['major_axis_length'][i] = temp

        properties['RL_diameter'] = properties['major_axis_length']
        properties['AP_diameter'] = properties['minor_axis_length']
        del properties['major_axis_length']
        del properties['minor_axis_length']

    # smooth the spinal cord shape with a gaussian kernel if required
    # TODO: not all properties can be smoothed
    if smooth_factor != 0.0:  # smooth_factor is in mm
        import scipy
        window = scipy.signal.hann(smooth_factor / np.mean(centerline.progressive_length))
        for property_name in property_list_local:
            properties[property_name] = scipy.signal.convolve(properties[property_name], window, mode='same') / np.sum(window)

    if compute_diameters:
        property_list_local.remove('major_axis_length')
        property_list_local.remove('minor_axis_length')
        property_list_local.append('RL_diameter')
        property_list_local.append('AP_diameter')
        property_list = property_list_local

    # Display properties on the referential space. Requires intervertebral disks
    if verbose == 2:
        x_increment = 'distance_from_C1'
        if fname_discs:
            x_increment = 'incremental_length'

        # Display the image and plot all contours found
        fig, axes = plt.subplots(len(property_list_local), sharex=True, sharey=False)
        for k, property_name in enumerate(property_list_local):
            axes[k].plot(properties[x_increment], properties[property_name])
            axes[k].set_ylabel(property_name)

        if fname_discs:
            properties['distance_disk_from_C1'] = centerline.distance_from_C1label  # distance between each disk and C1 (or first disk)
            xlabel_disks = [centerline.convert_vertlabel2disklabel[label] for label in properties['distance_disk_from_C1']]
            xtick_disks = [properties['distance_disk_from_C1'][label] for label in properties['distance_disk_from_C1']]
            plt.xticks(xtick_disks, xlabel_disks, rotation=30)
        else:
            axes[-1].set_xlabel('Position along the spinal cord (in mm)')

        plt.show()

    # Removing temporary folder
    os.chdir(curdir)
    if remove_temp_files:
        sct.rmtree(path_tmp)

    return property_list, properties


"""
Example of script that averages spinal cord shape from multiple subjects/patients, in a common reference frame (PAM50)
def prepare_data():

    folder_dataset = '/Volumes/data_shared/sct_testing/large/'
    import isct_test_function
    import json
    json_requirements = 'gm_model=0'
    data_subjects, subjects_name = sct_pipeline.generate_data_list(folder_dataset, json_requirements=json_requirements)

    fname_seg_images = []
    fname_discss = []
    group_images = []

    for subject_folder in data_subjects:
        if os.path.exists(os.path.join(subject_folder, 't2')):
            if os.path.exists(os.path.join(subject_folder, 't2', 't2_seg_manual.nii.gz')) and os.path.exists(os.path.join(subject_folder, 't2', 't2_disks_manual.nii.gz')):
                fname_seg_images.append(os.path.join(subject_folder, 't2', 't2_seg_manual.nii.gz'))
                fname_discss.append(os.path.join(subject_folder, 't2', 't2_disks_manual.nii.gz'))
                json_file = io.open(os.path.join(subject_folder, 'dataset_description.json'))
                dic_info = json.load(json_file)
                json_file.close()
                # pass keys and items to lower case
                dic_info = dict((k.lower(), v.lower()) for k, v in dic_info.items())
                if dic_info['pathology'] == 'HC':
                    group_images.append('b')
                else:
                    group_images.append('r')

    sct.printv('Number of images', len(fname_seg_images))

    property_list = ['area',
                     'equivalent_diameter',
                     'ratio_major_minor',
                     'eccentricity',
                     'solidity']

    average_properties(fname_seg_images, property_list, fname_discss, group_images, verbose=1)

"""
