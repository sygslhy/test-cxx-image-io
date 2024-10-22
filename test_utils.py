from pathlib import Path

import numpy as np
import pytest

from cxx_image_io import (ImageLayout, ImageMetadata, PixelRepresentation,
                          PixelType)
from cxx_image_io import split_image_channels

bayer_array = np.array([[1, 2] * 10 + [3, 4] * 10] * 10,
                       dtype=np.uint16).reshape(20, 20)
bayer_metadata = ImageMetadata()
bayer_metadata.fileInfo.width = 20
bayer_metadata.fileInfo.height = 20
bayer_metadata.fileInfo.pixelType = PixelType.BAYER_RGGB
bayer_metadata.fileInfo.imageLayout = ImageLayout.PLANAR
bayer_metadata.fileInfo.pixelRepresentation = PixelRepresentation.UINT16

rgb_array = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                      [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                      [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                      [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]])

rgb_metadata = ImageMetadata()
rgb_metadata.fileInfo.width = 4
rgb_metadata.fileInfo.height = 4
rgb_metadata.fileInfo.pixelType = PixelType.RGB
rgb_metadata.fileInfo.imageLayout = ImageLayout.INTERLEAVED
rgb_metadata.fileInfo.pixelRepresentation = PixelRepresentation.UINT8

rgb_planar_array = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]],
                             [[3, 3], [3, 3]]])

rgb_planar_metadata = ImageMetadata()
rgb_planar_metadata.fileInfo.width = 2
rgb_planar_metadata.fileInfo.height = 2
rgb_planar_metadata.fileInfo.pixelType = PixelType.RGB
rgb_planar_metadata.fileInfo.imageLayout = ImageLayout.PLANAR
rgb_planar_metadata.fileInfo.pixelRepresentation = PixelRepresentation.UINT8

yuv_array = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                      [2, 2, 2, 2], [3, 3, 3, 3]])

yuv_metadata = ImageMetadata()
yuv_metadata.fileInfo.width = 4
yuv_metadata.fileInfo.height = 4
yuv_metadata.fileInfo.pixelType = PixelType.YUV
yuv_metadata.fileInfo.imageLayout = ImageLayout.YUV_420
yuv_metadata.fileInfo.pixelRepresentation = PixelRepresentation.UINT8

nv12_array = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                       [2, 3, 2, 3], [2, 3, 2, 3]])

nv12_metadata = ImageMetadata()
nv12_metadata.fileInfo.width = 4
nv12_metadata.fileInfo.height = 4
nv12_metadata.fileInfo.pixelType = PixelType.YUV
nv12_metadata.fileInfo.imageLayout = ImageLayout.NV12
nv12_metadata.fileInfo.pixelRepresentation = PixelRepresentation.UINT8


@pytest.mark.parametrize('array, metadata, ref_values', [
    (bayer_array, bayer_metadata, (1, 2, 3, 4)),
    (rgb_array, rgb_metadata, (1, 2, 3)),
    (rgb_planar_array, rgb_planar_metadata, (1, 2, 3)),
    (yuv_array, yuv_metadata, (
        1,
        2,
        3,
    )),
    (nv12_array, nv12_metadata, (1, 2, 3)),
])
def test_split_channels(array, metadata, ref_values):
    channels = split_image_channels(array, metadata)
    if metadata.fileInfo.pixelType == PixelType.BAYER_RGGB:
        w, h = metadata.fileInfo.width // 2, metadata.fileInfo.height // 2
        assert channels['r'].shape == (h, w) and np.all(channels['r'] == 1)
        assert channels['gr'].shape == (h, w) and np.all(channels['gr'] == 2)
        assert channels['gb'].shape == (h, w) and np.all(channels['gb'] == 3)
        assert channels['b'].shape == (h, w) and np.all(channels['b'] == 4)
    elif metadata.fileInfo.pixelType == PixelType.RGB:
        w, h = metadata.fileInfo.width, metadata.fileInfo.height
        assert channels['r'].shape == (h, w) and np.all(channels['r'] == 1)
        assert channels['g'].shape == (h, w) and np.all(channels['g'] == 2)
        assert channels['b'].shape == (h, w) and np.all(channels['b'] == 3)
    elif metadata.fileInfo.pixelType == PixelType.YUV:
        w, h = metadata.fileInfo.width, metadata.fileInfo.height
        sampled_w, sampled_h = w // 2, h // 2
        assert channels['y'].shape == (h, w) and np.all(channels['y'] == 1)
        assert channels['u'].shape == (sampled_h, sampled_w) and np.all(
            channels['u'] == 2)
        assert channels['v'].shape == (sampled_h, sampled_w) and np.all(
            channels['v'] == 3)
