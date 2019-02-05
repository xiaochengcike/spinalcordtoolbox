from __future__ import absolute_import

import os
import numpy as np
import nibabel as nib
import sct_utils as sct
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.image as msct_image

from spinalcordtoolbox.deepseg_sc import core as deepseg_sc


def _create_fake_t2_sc(input_size):
    data = np.zeros(input_size)
    xx, yy = np.mgrid[:input_size[0], :input_size[1]]
    circle = (xx - input_size[0] // 2) ** 2 + (yy - input_size[1] // 2) ** 2

    for zz in range(data.shape[2]):
        data[:,:,zz] += np.logical_and(circle < 400, circle >= 200) * 250 # CSF
        data[:,:,zz] += (circle < 200) * 70 # SC
        # Note: values of the fake SC and CSF have been chosen by looking at the normalized intensities of im_norm_in (cf deepseg_sc/core)

    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())

    return img


def test_segment_2d():
    contrast_test = 't2'
    model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc.h5'.format(contrast_test))   

    img = _create_fake_t2_sc((48,48,1))
    seg = deepseg_sc.segment_2d(model_path, contrast_test, (48, 48), img)
    
    seg_im = msct_image.zeros_like(img)
    seg_gt_im = seg_im.copy()
    seg_gt_im.data = (img.data == 70)
    seg_im.data = seg

    assert np.any(seg[img.data == 70]) == True  # check if SC detected
    assert np.any(seg[img.data != 70]) == False  # check if no FP
    assert msct_image.compute_dice(seg_im, seg_gt_im) > 0.80
