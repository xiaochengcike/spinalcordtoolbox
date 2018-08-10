#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.aggregate_slicewise

from msct_image import Image
import numpy as np
import nibabel as nib
import pytest
import msct_image
from spinalcordtoolbox import aggregate_slicewise

@pytest.fixture(scope="session")
def dummy_vert_level():
    """Create a dummy image representing vertebral labeling."""
    nx, ny, nz = 9, 9, 9  # image dimension
    data = np.zeros((nx, ny, nz))
    # define vertebral level for each slice as a pixel at the center of the image
    data[4, 4, :] = [2, 2, 2, 3, 3, 3, 4, 4, 4]
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = msct_image.Image(nii.get_data(), hdr=nii.header,
                           orientation="RPI",
                           dim=nii.header.get_data_shape(),
                           )
    return img


# noinspection 801,PyShadowingNames
def test_aggregate_per_level(dummy_vert_level):
    """Test extraction of metrics aggregation per vertebral level"""
    group_funcs = (('mean', np.mean), ('std', np.std))
    metrics = {'metric1': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(metrics, slices=[], levels=[2, 3, 4], perslice=False,
                                                                   perlevel=True, im_vert_level=dummy_vert_level,
                                                                   group_funcs=group_funcs)
    assert agg_metrics == {'metric1':
                               {(0, 1, 2): {'std': 0.81649658092772603, 'VertLevel': (2,), 'mean': 2.0},
                                (3, 4, 5): {'std': 0.81649658092772603, 'VertLevel': (3,), 'mean': 5.0},
                                (6, 7, 8): {'std': 0.81649658092772603, 'VertLevel': (4,), 'mean': 8.0}}}


# noinspection 801,PyShadowingNames
def test_aggregate_across_level(dummy_vert_level):
    """Test extraction of metrics aggregation across vertebral levels"""
    group_funcs = (('mean', np.mean), ('std', np.std))
    metrics = {'metric1': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(metrics, slices=[], levels=[2, 3, 4], perslice=False,
                                                                   perlevel=False, im_vert_level=dummy_vert_level,
                                                                   group_funcs=group_funcs)
    assert agg_metrics == {'metric1':
                               {(0, 1, 2, 3, 4, 5, 6, 7, 8):
                                    {'std': 2.5819888974716112, 'VertLevel': (2, 3, 4), 'mean': 5.0}}}