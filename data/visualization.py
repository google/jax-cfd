# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualization utilities."""

from typing import Any, BinaryIO, Callable, Optional, List, Tuple, Union
from jax_cfd.base import grids
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as Image
import seaborn as sns


NormFn = Callable[[grids.Array, int], mpl.colors.Normalize]


def quantile_normalize_fn(
    image_data: grids.Array,
    image_num: int,
    quantile: float = 0.999
) -> mpl.colors.Normalize:
  """Returns `mpl.colors.Normalize` object that range defined by data quantile.

  Args:
    image_data: data for which `Normalize` object is produced.
    image_num: number of frame in the series. Not used.
    quantile: quantile that should be included in the range.

  Returns:
    `mpl.colors.Normalize` that covers the range of values that include quantile
    of `image_data` values.
  """
  del image_num  # not used by `quantile_normalize_fn`.
  max_to_include = np.quantile(abs(image_data), quantile)
  norm = mpl.colors.Normalize(vmin=-max_to_include, vmax=max_to_include)
  return norm


def resize_image(
    image: Image.Image,
    longest_side: int,
    resample: int = Image.NEAREST,
) -> Image.Image:
  """Resize an image, preserving its aspect ratio."""
  resize_factor = longest_side / max(image.size)
  new_size = tuple(round(s * resize_factor) for s in image.size)
  return image.resize(new_size, resample)


def trajectory_to_images(
    trajectory: grids.Array,
    compute_norm_fn: NormFn = quantile_normalize_fn,
    cmap: mpl.colors.ListedColormap = sns.cm.icefire,
    longest_side: Optional[int] = None,
) -> List[Image.Image]:
  """Converts scalar trajectory with leading time axis into a list of images."""
  images = []
  for i, image_data in enumerate(trajectory):
    norm = compute_norm_fn(image_data, i)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    img = Image.fromarray(mappable.to_rgba(image_data, bytes=True))
    if longest_side is not None:
      img = resize_image(img, longest_side)
    images.append(img)
  return images


# TODO(dkochkov) consider generalizing this to a general facet.
def horizontal_facet(
    separate_images: List[List[Image.Image]],
    relative_separation_width: float,
    separation_rgb: Tuple[int, int, int] = (255, 255, 255)
) -> List[Image.Image]:
  """Stitches separate images into a single one with a separation strip.

  Args:
    separate_images: lists of images each representing time series. All images
      must have the same size.
    relative_separation_width: width of the separation defined as a fraction of
      a separate image.
    separation_rgb: rgb color code of the separation strip to add between
      adjacent images.

  Returns:
    list of merged images that contain images passed as `separate_images` with
    a separating strip.
  """
  images = []
  for frames in zip(*separate_images):
    images_to_combine = len(frames)
    separation_width = round(frames[0].width * relative_separation_width)
    image_height = frames[0].height
    image_width = (frames[0].width * images_to_combine +
                   separation_width * (images_to_combine - 1))
    full_im = Image.new('RGB', (image_width, image_height))

    sep_im = Image.new('RGB', (separation_width, image_height), separation_rgb)
    full_im = Image.new('RGB', (image_width, image_height))

    width_offset = 0
    height_offset = 0
    for frame in frames:
      full_im.paste(frame, (width_offset, height_offset))
      width_offset += frame.width
      if width_offset < full_im.width:
        full_im.paste(sep_im, (width_offset, height_offset))
        width_offset += sep_im.width
    images.append(full_im)
  return images


def save_movie(
    images: List[Image.Image],
    output_path: Union[str, BinaryIO],
    duration: float = 150.,
    loop: int = 0,
    **kwargs: Any
):
  """Saves `images` as a movie of duration `duration` to `output_path`.

  Args:
    images: list of images representing time series that will be saved as movie.
    output_path: file handle or cns path to where save the movie.
    duration: duration of the movie in milliseconds.
    loop: number of times to loop the movie. 0 interpreted as indefinite.
    **kwargs: additional keyword arguments to be passed to `Image.save`.
  """
  images[0].save(output_path, save_all=True, append_images=images[1:],
                 duration=duration, loop=loop, **kwargs)
