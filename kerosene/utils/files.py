# -*- coding: utf-8 -*-
# Copyright 2019 Kerosene Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from glob import glob
from typing import List, Tuple


def split_filename(filepath: str) -> Tuple[str, str, str]:
    """
    Split a filepath into the directory, base, and extension

    Args:
        filepath (str): The base file path.

    Returns:
        Tuple: The complete file path, base path and file extension.
    """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def extract_file_paths(path: str, ext='*.nii*') -> List[str]:
    """
    Grab all `ext` files in a directory and sort them for consistency.

    Args:
        path (str): File path.
        ext (str): File's extension to grab.

    Returns:
        list: A list of string containing every file paths.
    """
    file_paths = sorted(glob(os.path.join(path, ext)))
    return file_paths


def should_create_dir(path, dir_name):
    return not os.path.exists(os.path.join(path, dir_name))
