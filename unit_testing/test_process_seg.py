#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.process_seg

import numpy as np
import nibabel as nib
import csv
import pytest
from spinalcordtoolbox import process_seg


@pytest.fixture(scope="session")
def dummy_segmentation():
    """Create a dummy image with a circle or ones running from top to bottom in the 3rd dimension"""
    # TODO: rotate the image to make sure the CSA properly accounts for the angle correction
    nx, ny, nz = 20, 20, 20  # image dimension
    fname_seg = 'dummy_segmentation.nii.gz'  # output seg
    data = np.random.random((nx, ny, nz))
    xx, yy = np.mgrid[:nx, :ny]
    # loop across slices and add a circle of radius 3 pixels
    for iz in range(nz):
        data[:, :, iz] = ((xx - nx/2) ** 2 + (yy - ny/2) ** 2 <= 3 ** 2) * 1
    xform = np.eye(4)
    img = nib.nifti1.Nifti1Image(data, xform)
    nib.save(img, fname_seg)
    return fname_seg


# noinspection 801,PyShadowingNames
def test_extract_centerline(dummy_segmentation):
    """Test extraction of centerline from input segmentation"""
    process_seg.extract_centerline(dummy_segmentation, 0, file_out='centerline')
    # open created csv file
    centerline_out = []
    with open('centerline.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()  # skip header
        for row in reader:
            centerline_out.append([int(i) for i in row])
    # build ground-truth centerline
    centerline_true = [[i, 9, 10] for i in range(20)]
    assert centerline_out == centerline_true


# noinspection 801,PyShadowingNames
def test_compute_csa(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation"""
    process_seg.compute_csa(dummy_segmentation, 1, 1, 1, '', '', fname_vertebral_labeling='', perslice=0, perlevel=0,
                            algo_fitting='hanning', type_window='hanning', window_length=10, angle_correction=True,
                            use_phys_coord=True, file_out='csa')
    # open created csv file
    with open('csa.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()  # skip header
        csa_out, angle_out = reader.next()[2:4]
    assert csa_out == '29.0'
    assert angle_out == '0.0'


# noinspection 801,PyShadowingNames
def test_compute_shape(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation"""
    process_seg.compute_shape(dummy_segmentation, 0, file_out='shape', overwrite=0, fname_discs='', verbose=1)
    # open created csv file
    # with open('shape.csv', 'rb') as f:
    #     reader = csv.reader(f)
    #     reader.next()  # skip header
    #     csa_out, angle_out = reader.next()[2:4]
    # assert csa_out == '29.0'
    # assert angle_out == '0.0'