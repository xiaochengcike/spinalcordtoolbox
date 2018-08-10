#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.aggregate_slicewise

from msct_image import Image
import numpy as np
import pytest
from spinalcordtoolbox import aggregate_slicewise

@pytest.fixture(scope="session")
def dummy_vert_level():
    """Create a dummy image representing vertebral labeling."""
    nx, ny, nz = 9, 9, 9  # image dimension
    data = np.zeros((nx, ny, nz))
    # define vertebral level for each slice as a pixel at the center of the image
    data[4, 4, :] = [2, 2, 2, 3, 3, 3, 4, 4, 4]
    im_vert_level = Image
    im_vert_level.data = data
    im_vert_level.dim = (im_vert_level.data.shape[0], im_vert_level.data.shape[1], im_vert_level.data.shape[2], 1,
                         1, 1, 1, 1)
    return im_vert_level


# noinspection 801,PyShadowingNames
def test_aggregate_metrics_by_vertebral_level(dummy_vert_level):
    """Test extraction of metrics aggregation across vertebral levels"""
    group_funcs = (('mean', np.mean), ('std', np.std))
    metrics = {'metric1': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    levels = [2, 3, 4]
    im_vert_levels = dummy_vert_level
    agg_metrics = aggregate_slicewise.aggregate_metrics_by_vertebral_level(metrics, im_vert_levels, levels,
                                                                           group_funcs=group_funcs)
    assert agg_metrics == {'metric1':
                               {2: {'std': 0.81649658092772603, 'mean': 2.0},
                                3: {'std': 0.81649658092772603, 'mean': 5.0},
                                4: {'std': 0.81649658092772603, 'mean': 8.0}}}