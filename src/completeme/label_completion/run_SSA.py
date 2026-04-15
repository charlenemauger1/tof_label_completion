import numpy as np
import pydicom
import os
import nibabel as nib
import torch
import matplotlib as plt
plt.use('TkAgg')
import scipy
import json
from pathlib import Path
import argparse
from loguru import logger
from distutils.dir_util import copy_tree
import time
import concurrent.futures
import re
from scipy.ndimage import map_coordinates

# Configuration
RV_LABEL = 3
RA_LABEL = 4
RV_MYO_LABEL = 8

LV_LABEL = 2
LA_LABEL = 5
LV_MYO_LABEL = 1

# @timing
def extract_info(path, dicom_format = True):
    if dicom_format:
        ds = pydicom.dcmread(path)
        data_array = ds.pixel_array.astype(np.float64)
        data_affine = np.eye(4)
        data_affine[:3, 0] = np.array(ds.ImageOrientationPatient[:3])
        data_affine[:3, 1] = np.array(ds.ImageOrientationPatient[3:])
        data_affine[:3, 2] = np.cross(data_affine[:3, 0], data_affine[:3, 1])
        # print(data_affine)
        data_affine[:3, 3] = np.array(ds.ImagePositionPatient)
        data_thickness = ds.SliceThickness

        return [data_array, data_affine, data_thickness]

    else:
        data = nib.load(path)
        data_array = data.get_fdata()
        data_affine = data.get_qform()
        data_thickness = data.header['pixdim'][3]

        return [data_array, data_affine, data_thickness]

# @timing
def intersection_resample(slice_a, slice_b):
    """
    calculate the intersection area by resampling slice_a (with thickness) into empty slice_b (same shape as slice_b)

    :slice_a (list): a list of data array, affine matrix and slice thickness for slice a
    :slice_b (list): a list of data array, affine matrix and slice thickness for slice b
    :return: a resampled intersection area in the empty slice_B
    :intersection_result (numpy array): 2D array of resampled intersection area (same shape as slice_b)
    """

    # read slice A and slice B information
    array_a = slice_a[0]
    affine_a = slice_a[1]
    thickness_a = 2
    array_b = slice_b[0]
    affine_b = slice_b[1]
    # thickness_b = 1

    # create block A with given thickness
    a_with_thickness = np.zeros((int(thickness_a), array_a.shape[0], array_a.shape[1]))
    a_with_thickness[:, ...] = array_a
    # affine matrix for block A
    affine_a_with_thickness = np.zeros(affine_a.shape)
    affine_a_with_thickness[affine_a != 0] = affine_a[affine_a != 0]
    affine_a_with_thickness[:, -1] -= affine_a_with_thickness[:, 2] * (a_with_thickness.shape[0]//2)

    # coordinate transformation, ijk to xyz
    # https://medium.com/redbrick-ai/dicom-coordinate-systems-3d-dicom-for-computer-vision-engineers-pt-1-61341d87485f
    ij_index = (np.array(np.meshgrid(np.arange(0, array_b.shape[1]), np.arange(0, array_b.shape[0]), indexing='ij')).T.reshape(array_b.shape[0], array_b.shape[1], 2))
    ijk_index = np.zeros((array_b.shape[0], array_b.shape[1], 4))
    ijk_index[:, :, :2] = ij_index
    # for matrix multiplication, make the last row as all 1
    ijk_index[:, :, -1] = 1
    # each entry for matrix ijk_index is in (i, j, 0, 1) format
    ijk_index = ijk_index.T.reshape(4, array_b.shape[0] * array_b.shape[1])
    # affine matrix for intersection area, which will be used for defining grid later
    affine_inter = np.dot(np.linalg.inv(affine_a_with_thickness), affine_b)
    # xyz coordinate system
    xyz_index_temp = np.dot(affine_inter, ijk_index)
    # remove the last row (all 1)
    xyz_index = xyz_index_temp[:3, :].reshape(3, array_b.shape[1], array_b.shape[0]).T

    # input tensor [N, C, D_in, H_in, W_in] and grid tensor [N, D_out, H_out, W_out, 3] for the grid_sampling function
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    input_tensor = torch.from_numpy(a_with_thickness.reshape((1, 1, a_with_thickness.shape[0], a_with_thickness.shape[1], a_with_thickness.shape[2])))
    grid = torch.from_numpy(xyz_index.reshape((1, 1, array_b.shape[0], array_b.shape[1], 3))).type(torch.DoubleTensor)

    # normalization to [-1, 1] for grid tensor
    norm_factor_0 = (input_tensor.shape[2] - 1) / 2
    norm_factor_1 = (input_tensor.shape[3] - 1) / 2
    norm_factor_2 = (input_tensor.shape[4] - 1) / 2

    grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - norm_factor_2) / (norm_factor_2 + 0.5)
    grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - norm_factor_1) / (norm_factor_1 + 0.5)
    grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - norm_factor_0) / (norm_factor_0 + 0.5)

    # resampled intersection area [N, C, D_out, H_out, W_out]
    intersection_result = torch.nn.functional.grid_sample(input_tensor, grid, mode='nearest', padding_mode='zeros',
                                                          align_corners=False)[0, 0, 0, ...].numpy()

    return intersection_result

def cal_target(moving_img, template_img):
    """
    Elementwise label match count across unique labels (excluding background 0).
    """
    labels = np.unique(moving_img)
    labels = labels[labels != 0]  # exclude background label 0

    target = 0
    for label in labels:
        match = (moving_img == label) & (template_img == label)
        target += np.count_nonzero(match)

    return target

def dice_score(moving_img, template_img):
    dice = np.zeros(5)
    for i in list(np.unique(moving_img)):
        if i != 0:
            dice_label = 2*np.sum(np.multiply((moving_img == i), (template_img == i)))/(np.sum(moving_img == i) + np.sum(template_img == i))
            dice[int(i)-1] = dice_label

    dice = dice/len(np.unique(moving_img))

    return dice

# @timing
def grid_search(moving_img, template_img, lr_unit, ud_unit):
    # define a grid for searching directions, where i>0 (right), i<0 (left), j>0 (up) and j<0 (down)
    direction = np.array(np.meshgrid(np.arange(0, lr_unit), np.arange(0, ud_unit))).T
    direction = direction - direction.shape[0] // 2
    # total number of searching times
    total_num_search = direction.shape[0] * direction.shape[1]
    direction = direction.reshape(total_num_search, 2)

    # a list of target value (element-wise multiplication in my case) for later optimization (maximization in my case)
    target_val_list = []

    for d in range(total_num_search):
        corrected_img = np.zeros(moving_img.shape)

        # right
        if direction[d][0] > 0:
            # up
            if direction[d][1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[d][1]),
                np.abs(direction[d][0]): moving_img.shape[1]] = moving_img[np.abs(direction[d][1]):moving_img.shape[0],
                                                               0: moving_img.shape[1] - np.abs(direction[d][0])]
                
                temp_target = cal_target(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))
            # down
            else:
                corrected_img[np.abs(direction[d][1]):moving_img.shape[0],
                np.abs(direction[d][0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[d][1]),
                                                               0: moving_img.shape[1] - np.abs(direction[d][0])]
                temp_target = cal_target(corrected_img, template_img)
                # temp_target = dice_score(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))
        # left
        else:
            # up
            if direction[d][1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[d][1]),
                0: moving_img.shape[1] - np.abs(direction[d][0])] = moving_img[np.abs(direction[d][1]):moving_img.shape[0],
                                                                   np.abs(direction[d][0]): moving_img.shape[1]]
                temp_target = cal_target(corrected_img, template_img)
                # temp_target = dice_score(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))
            # down
            else:
                corrected_img[np.abs(direction[d][1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[d][0])] = moving_img[
                                                                   0: moving_img.shape[0] - np.abs(direction[d][1]),
                                                                   np.abs(direction[d][0]): moving_img.shape[1]]
                temp_target = cal_target(corrected_img, template_img)
                # temp_target = dice_score(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))

    best_direction = max(target_val_list, key=lambda x: x[1])

    return best_direction

def apply_affine_transformation(path_data, dicom_file_name, direction, dicom_format = True):
    if dicom_format:
        path_dicom = os.path.join(path_data, dicom_file_name)
        dataset = pydicom.dcmread(path_dicom)
        moving_img = dataset.pixel_array.astype(np.float64)
        corrected_img = np.zeros(moving_img.shape)

        # slice shifting by changing the content for the slice
        # right
        if direction[0] > 0:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

        # left
        else:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                 np.abs(direction[0]): moving_img.shape[1]]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                 0: moving_img.shape[0] - np.abs(direction[1]),
                                                                 np.abs(direction[0]): moving_img.shape[1]]

        # save new dicom file with same header information as before
        moving_img = np.short(corrected_img)
        dataset.PixelData = moving_img.tobytes()
        dataset.save_as(path_dicom)
    else:
        path_dicom = os.path.join(path_data, dicom_file_name)
        data = nib.load(path_dicom)
        moving_img =data.get_fdata()[..., 0].T
        corrected_img = np.zeros(moving_img.shape)

        # slice shifting by changing the content for the slice
        # right
        if direction[0] > 0:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

        # left
        else:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                 np.abs(direction[0]): moving_img.shape[1]]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                 0: moving_img.shape[0] - np.abs(direction[1]),
                                                                 np.abs(direction[0]): moving_img.shape[1]]

        # save new dicom file with same header information as before
        moving_img_new = corrected_img.T
        new_nifti = nib.Nifti1Image(moving_img_new[:, :, np.newaxis].astype(np.uint8), data.affine)
        nib.save(new_nifti, path_dicom)

    return

def apply_affine_transformation_data(original_data, direction):
    moving_img = original_data
    corrected_img = np.zeros(moving_img.shape)

    # slice shifting by changing the content for the slice
    # right
    if direction[0] > 0:
        # up
        if direction[1] > 0:
            corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
            np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                            0: moving_img.shape[1] - np.abs(direction[0])]

        # down
        else:
            corrected_img[np.abs(direction[1]):moving_img.shape[0],
            np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                            0: moving_img.shape[1] - np.abs(direction[0])]

    # left
    else:
        # up
        if direction[1] > 0:
            corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
            0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                np.abs(direction[0]): moving_img.shape[1]]

        # down
        else:
            corrected_img[np.abs(direction[1]):moving_img.shape[0],
            0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                0: moving_img.shape[0] - np.abs(direction[1]),
                                                                np.abs(direction[0]): moving_img.shape[1]]

    return corrected_img

def find_max_and_indices(arr):
    first_dim_half = arr.shape[0] // 2
    second_dim_half = arr.shape[1] // 2

    # Define the slice range
    i_start = first_dim_half - 10
    i_end = first_dim_half + 10
    j_start = second_dim_half - 10
    j_end = second_dim_half + 10

    # Extract the subarray
    sub_arr = arr[i_start:i_end, j_start:j_end]

    # Find the maximum value
    max_value = np.max(sub_arr)

    # Get all indices of the maximum value in the subarray
    rel_indices = np.argwhere(np.abs(sub_arr - max_value) < 1e-5)

    # Convert subarray-relative indices to original array indices
    max_indices = [(i + i_start, j + j_start) for i, j in rel_indices]

    max_indices = [(int(x), int(y)) for x, y in max_indices]

    return max_value, max_indices


def fft_search_optimized(template_img, moving_img):
    template_img_f = template_img.astype(np.float32)
    moving_img_f = moving_img.astype(np.float32)

    fft_ref = scipy.signal.correlate(template_img_f, template_img_f, method='fft', mode='full')

    unique_labels = np.unique(moving_img)
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return (0, 0)

    s1 = template_img_f.shape
    s2 = moving_img_f.shape
    out_shape = tuple(np.array(s1) + np.array(s2) - 1)

    sum_fft_product_freq = np.zeros(out_shape, dtype=np.complex64)

    for label in unique_labels:
        template_binary = (template_img_f == label).astype(np.float32)
        moving_binary = (moving_img_f == label).astype(np.float32)

        fft_template_binary = scipy.fft.fft2(template_binary, s=out_shape)
        fft_moving_binary = scipy.fft.fft2(moving_binary, s=out_shape)

        sum_fft_product_freq += fft_template_binary * np.conjugate(fft_moving_binary)

    fft_new_raw = scipy.fft.ifft2(sum_fft_product_freq).real

    fft_new_shifted = scipy.fft.fftshift(fft_new_raw)
    peak_index = find_max_and_indices(fft_new_shifted)[1]

    center_r, center_c = np.array(fft_new_shifted.shape) // 2

    peak_index_ref = find_max_and_indices(fft_ref)[1]
    
    # Handle the unlikely but possible case where peak_index_ref might be empty
    if len(peak_index_ref) == 0:
        return (0, 0)

    # Ensure ref_actual_shift is also composed of Python ints if it matters for comparison later
    ref_actual_shift_np = peak_index_ref[0] - np.array([s1[0]-1, s1[1]-1])
    ref_actual_shift = (int(ref_actual_shift_np[0]), int(ref_actual_shift_np[1]))


    max_shift_abs = 20
    best_direction = (0, 0) # Initialize with standard Python ints

    for current_index_shifted in peak_index:
        # current_actual_shift will be a numpy array of np.int64
        current_actual_shift_np = current_index_shifted - np.array([center_r, center_c])

        # Cast these components to Python ints immediately
        current_actual_shift_r = int(current_actual_shift_np[0])
        current_actual_shift_c = int(current_actual_shift_np[1])

        tmp_dir_row_diff = current_actual_shift_r - ref_actual_shift[0]
        tmp_dir_col_diff = current_actual_shift_c - ref_actual_shift[1]

        # tmp_dir_row_diff and tmp_dir_col_diff should now be Python ints if the above cast worked.
        # However, to be absolutely certain, re-cast explicitly for tmp_dir
        tmp_dir = (int(tmp_dir_col_diff), int(-tmp_dir_row_diff))

        abs_diff = abs(tmp_dir[0]) + abs(tmp_dir[1])
        if abs_diff == 0:
            return (0, 0) 
        elif abs_diff < max_shift_abs:
            max_shift_abs = abs_diff
            best_direction = tmp_dir # best_direction gets assigned a tuple of Python ints here

    # This return should now be a tuple of Python ints
    return best_direction

def fft_search(template_img, moving_img):
    fft_new = None
    label_list = np.unique(moving_img)

    for l in range(1, len(label_list)):
        label = label_list[l]
        moving_binary = (moving_img == label).astype(np.int32)
        template_binary = (template_img == label).astype(np.int32)
        tmp_fft = scipy.signal.correlate(template_binary, moving_binary, method='fft')
        if fft_new is None:
            fft_new = tmp_fft
        else:
            fft_new += tmp_fft

    fft_ref = scipy.signal.correlate(template_img, template_img, method='fft')

    peak_index_ref = find_max_and_indices(fft_ref)[1]
    peak_index = find_max_and_indices(fft_new)[1]

    max_shift_abs = 20
    best_direction = (0, 0)
    ref_index = peak_index_ref[0]  # use the first peak index as reference

    for current_index in peak_index:
        tmp_dir = (current_index[1] - ref_index[1], ref_index[0] - current_index[0])
        abs_diff = abs(tmp_dir[0]) + abs(tmp_dir[1])
        if abs_diff == 0:
            return tmp_dir
        elif abs_diff < max_shift_abs:
            max_shift_abs = abs_diff
            best_direction = tmp_dir

    return best_direction

def count_values(arr):
   unique, counts = np.unique(arr, return_counts=True)
   return dict(zip(unique, counts))

def calculate_shift(path_data: Path, case_name : str):
    dicom_list = sorted(os.listdir(path_data))
    path_dicom = path_data
    num_iter = 5
    ssa_hist = {}

    for key in dicom_list:
        ssa_hist[key] = [0, 0]
    for j in range(num_iter):
        logger.info(f'----------------- Iteration {j+1} for {case_name} -----------------')
        # iterate over dicom file under each data, start with LAX-4CH
        for k in range(len(dicom_list)):
            moving_img_name = dicom_list[k]
            # remove current slice for later intersection calculation (will add back later)
            dicom_list.remove(moving_img_name)
            moving_img_list = extract_info(os.path.join(path_dicom, moving_img_name), dicom_format=False)
            moving_img_list[0] = moving_img_list[0][...,0].T
            template_img = np.zeros(moving_img_list[0].shape)
            # we need delete the slice (like apex) which contains less information
            if len(np.unique(moving_img_list[0])) != 1:
                for s in range(len(dicom_list)):
                    tmp_slice_name = dicom_list[s]
                    tmp_slice_list = extract_info(os.path.join(path_dicom, tmp_slice_name), dicom_format=False)
                    tmp_slice_list[0] = tmp_slice_list[0][...,0].T
                    single_intersection = intersection_resample(tmp_slice_list, moving_img_list)
                    template_img = np.maximum(template_img, single_intersection)
                try:
                    ratio = count_values(template_img)[1.0] / count_values(template_img)[2.0]
                    num_ones = count_values(template_img)[1.0]
                except:
                    logger.info('This slice does not cut through LV :' + os.path.join(path_dicom, moving_img_name))
                    ratio = 0
                    num_ones = 0

                if (('sa' in moving_img_name.lower()) and (len(np.unique(template_img)) == 6) and (len(np.unique(moving_img_list[0])) < 3)) or (
                        ('sa' in moving_img_name.lower()) and (ratio < 0.2) and (num_ones < 5)):
                    moving_img_list_updated = extract_info(os.path.join(path_dicom, moving_img_name), dicom_format=False)
                    moving_img_list_updated[0] = np.zeros(moving_img_list_updated[0].shape)
                    new_nifti = nib.Nifti1Image(moving_img_list_updated[0].astype(np.uint8), moving_img_list_updated[1])

                    nib.save(new_nifti, os.path.join(path_dicom, moving_img_name))
                    #count += 1
                    logger.error('deleted slice location:' + os.path.join(path_dicom, moving_img_name))
                else:
                    in_plane_shift = fft_search_optimized(template_img, moving_img_list[0])
                    ssa_hist[moving_img_name][0] += in_plane_shift[0]
                    ssa_hist[moving_img_name][1] += in_plane_shift[1]
                    logger.info('iteration:' + str(j+1) + '; slice name:' + moving_img_name + '; current shift:' + str(in_plane_shift))
                    # update dicom file
                    apply_affine_transformation(path_dicom, moving_img_name, in_plane_shift, False)
            # add back the slice name

            dicom_list.append(moving_img_name)

        logger.info('iteration' + str(j+1) + ' finished')

    return ssa_hist

def apply_shift(path_data: Path, json_path: Path):
    with open(json_path, 'r') as file:
        ssa_shift = json.load(file)

    key_set = set(ssa_shift.keys())  # faster lookups
    dicom_list_full = sorted(os.listdir(path_data))  # don't modify this list

    logger.info('----------------------------------------------------------')
    for moving_img_name in dicom_list_full:
        moving_img_path = os.path.join(path_data, moving_img_name)
        moving_img_list = extract_info(moving_img_path, dicom_format=False)
        moving_img_list[0] = moving_img_list[0][..., 0].T

        # Replace frame number with f1 (reference frame)
        ref_slice_name = re.sub(r'f\d+', 'f001', moving_img_name)
        if ref_slice_name in key_set:
            in_plane_shift = ssa_shift[ref_slice_name]
            apply_affine_transformation(path_data, moving_img_name, in_plane_shift, False)
        else:
            # Replace image with zero array
            moving_img_list_updated = extract_info(moving_img_path, dicom_format=False)
            moving_img_list_updated[0] = np.zeros_like(moving_img_list_updated[0])
            new_nifti = nib.Nifti1Image(moving_img_list_updated[0].astype(np.uint8), moving_img_list_updated[1])
            nib.save(new_nifti, moving_img_path)
            logger.info(f"deleted slices: {os.path.basename(path_data)}_{moving_img_name}")

def remove_rv_in_la(folder, output_folder):
    # Load all segmentations once
    seg_files = {f: nib.load(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith((".nii", ".nii.gz"))}

    # Identify 4Ch file
    fourch_file = next((f for f in seg_files if '4Ch' in f or '4CH' in f), None)
    if not fourch_file:
        raise RuntimeError("4Ch segmentation not found.")

    fourch_img = seg_files[fourch_file]
    fourch_data = fourch_img.get_fdata()
    affine_4ch = fourch_img.affine
    inv_affine_4ch = np.linalg.inv(affine_4ch)

    for fname, sax_img in seg_files.items():
        if 'SA' not in fname or fname == fourch_file:
            continue

        sax_data = sax_img.get_fdata()
        affine_sax = sax_img.affine
        corrected_data = sax_data.copy()

        # Find all RV voxels in SAX
        rv_voxels = np.argwhere(sax_data == RV_LABEL)
        if rv_voxels.size > 0:
            rv_voxels_h = np.c_[rv_voxels, np.ones(len(rv_voxels))]
            world_coords_rv = (affine_sax @ rv_voxels_h.T).T[:, :3]
            vox_4ch_rv = (inv_affine_4ch @ np.c_[world_coords_rv, np.ones(len(world_coords_rv))].T).T[:, :3].T
            sampled_labels_rv = map_coordinates(fourch_data, vox_4ch_rv, order=0, mode='nearest')
            intersecting_indices_ra = sampled_labels_rv == RA_LABEL
            num_overlap_ra = np.sum(intersecting_indices_ra)
        else:
            num_overlap_ra = 0

        # Find all LV voxels in SAX
        lv_voxels = np.argwhere(sax_data == LV_LABEL)
        if lv_voxels.size > 0:
            lv_voxels_h = np.c_[lv_voxels, np.ones(len(lv_voxels))]
            world_coords_lv = (affine_sax @ lv_voxels_h.T).T[:, :3]
            vox_4ch_lv = (inv_affine_4ch @ np.c_[world_coords_lv, np.ones(len(world_coords_lv))].T).T[:, :3].T
            sampled_labels_lv = map_coordinates(fourch_data, vox_4ch_lv, order=0, mode='nearest')
            intersecting_indices_la = sampled_labels_lv == LA_LABEL
            intersecting_indices_rv = sampled_labels_lv == RV_LABEL
            num_overlap_la = np.sum(intersecting_indices_la)
            num_overlap_rv = np.sum(intersecting_indices_rv)
        else:
            num_overlap_la = 0
            num_overlap_rv = 0

        if num_overlap_ra > 0:
            corrected_data[corrected_data == RV_LABEL] = 0
            corrected_data[corrected_data == RV_MYO_LABEL] = 0

        if num_overlap_la > 0:
            corrected_data[corrected_data == LV_LABEL] = 0
            corrected_data[corrected_data == LV_MYO_LABEL] = 0

        if num_overlap_rv > 0:
            corrected_data[corrected_data == LV_LABEL] = 0
            corrected_data[corrected_data == LV_MYO_LABEL] = 0
            corrected_data[corrected_data == RV_LABEL] = 0
            corrected_data[corrected_data == RV_MYO_LABEL] = 0

        if (num_overlap_ra + num_overlap_la + num_overlap_rv) == 0:
            continue

        corrected_img = nib.Nifti1Image(corrected_data.astype(np.uint8), sax_img.affine, sax_img.header)
        corrected_path = os.path.join(output_folder, fname)
        nib.save(corrected_img, corrected_path)

def process_single_patient_folder(case_path: Path, output_base_path: Path, args, log_enabled: bool):
    """
    Processes a single patient folder (either an individual subfolder or one from a main folder).
    """
    logger.info(f"Processing patient: {case_path.name}")
    patient_output_folder = output_base_path / case_path.name
    patient_output_folder.mkdir(parents=True, exist_ok=True)

    if args.step == 'calculate':
        # Ensure the ED frame folder exists in the input
        ed_frame_source_path = case_path / f'{case_path.name}_{args.ed_frame:03}'
        if not ed_frame_source_path.is_dir():
            logger.error(f"ED frame folder not found for {case_path.name} at {ed_frame_source_path}. Skipping 'calculate' step for this patient.")
            return

        # Copy the ED frame data to the output path
        ed_frame_dest_path = patient_output_folder / f'{case_path.name}_{args.ed_frame:03}'
        copy_tree(str(ed_frame_source_path), str(ed_frame_dest_path))
        
        # Calculate shifts
        shifts = calculate_shift(ed_frame_dest_path, case_path.name)

        # Apply RV/LA removal
        remove_rv_in_la(ed_frame_dest_path, ed_frame_dest_path) # operate in place
        
        # Save shifts to JSON
        json_file_path = patient_output_folder / f'{case_path.name}_translation_file.json'
        with open(json_file_path, 'w') as f:
            json.dump(shifts, f)
        logger.info(f"Calculated shifts saved to {json_file_path}")

    if args.step == 'infer':
        json_file_path = patient_output_folder / f'{case_path.name}_translation_file.json'
        if not json_file_path.exists():
            logger.error(f"Cannot find shifts file for {case_path.name}: {json_file_path} does not exist. Skipping 'infer' step for this patient.")
            return
        
        # Get all frame folders for this patient
        time_frame_folders = [f for f in case_path.iterdir() if f.is_dir() and re.match(f'{case_path.name}_\d+', f.name, re.IGNORECASE)]
        
        frame_numbers = []
        for folder in time_frame_folders:
            match = re.search(f'{case_path.name}_*(\d+)', folder.name, re.IGNORECASE)
            if match:
                frame_numbers.append(int(match[1]))
        frame_numbers = sorted(list(set(frame_numbers))) # Get unique sorted frame numbers

        for frame in frame_numbers:
            if frame != args.ed_frame: # Skip ED frame if it's already processed/calculated
                frame_source_path = case_path / f'{case_path.name}_{frame:03}'
                if not frame_source_path.is_dir():
                    logger.warning(f"Frame folder {frame_source_path} not found for {case_path.name}. Skipping this frame.")
                    continue

                frame_dest_path = patient_output_folder / f'{case_path.name}_{frame:03}'
                copy_tree(str(frame_source_path), str(frame_dest_path))
                logger.info(f"Applying shifts for {case_path.name} frame {frame:03}")
                apply_shift(frame_dest_path, json_file_path)
                remove_rv_in_la(frame_dest_path, frame_dest_path) # operate in place


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='This script performs breath-hold misregistration correction by calculating the corresponding shift from the ED frame.',
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )

    # Create a mutually exclusive group for input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--input_main_folder", 
        type=Path,
        help="Path to the main input folder containing N patient subfolders (e.g., patient IDs).\n"
             "The script will process each subfolder in parallel if multiprocessing is enabled."
    )
    group.add_argument(
        "-s", "--input_subfolder", 
        type=Path,
        help="Path to a single patient subfolder to process (e.g., a specific patient ID).\n"
             "Multiprocessing will be disabled for this option."
    )
    parser.add_argument('-o', '--output_path', type=Path,
                        help='Path where to save the applied translations json file and corrected nifitis', 
                        default='./corrected_niftis')
    parser.add_argument('-ed', '--ed_frame', type=int,
                        help='ED frame number to use as reference', default=1)
    parser.add_argument('-step', '--step', type=str,
                        choices=['calculate', 'infer'],
                        help='Defines the processing step:\n'
                             '"calculate": Calculate shifts for the ED frame and save to JSON.\n'
                             '"infer": Apply calculated shifts from JSON to remaining frames.',
                        default='calculate')   
    parser.add_argument("--log", action="store_true", help="Enable detailed logging to a file and console.")
    parser.add_argument("--no_multiprocessing", action="store_true", 
                        help="Disable multiprocessing even when processing multiple patient folders. "
                             "Ignored if --input_subfolder is used (multiprocessing is always off then).")
    parser.add_argument('--max_workers', type=int, default=4, help='Max number of worker threads') 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    assert args.output_path.parent.exists(), \
        f'Cannot create output directory. Parent folder of {args.output_path} does not exist.'
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    if args.log:
        logger.add(str(args.output_path / "misregistration_log.log"), rotation="5 MB", level="INFO", enqueue=True)
        logger.info("Logging enabled.")
    else:
        logger.remove() # Remove default handler
        # To completely silence loguru:
        logger.configure(handlers=[{"sink": lambda msg: None}]) 
        logger.info("Logging disabled. No messages will be shown or saved.")

    # Determine input folders based on arguments
    patient_folders_to_process = []
    use_multiprocessing = False

    if args.input_main_folder:
        input_base_path = args.input_main_folder
        if not input_base_path.is_dir():
            logger.error(f"Error: The provided main input folder '{input_base_path}' is not a valid directory.")
            exit(1)
        for subfolder in input_base_path.iterdir():
            if subfolder.is_dir():
                patient_folders_to_process.append(subfolder)
        use_multiprocessing = not args.no_multiprocessing # Enable if not explicitly disabled
        logger.info(f"Found {len(patient_folders_to_process)} patient folders in main input: {input_base_path}")
        if use_multiprocessing:
            logger.info("Multiprocessing enabled for patient folders.")
        else:
            logger.info("Multiprocessing disabled for patient folders (due to --no_multiprocessing flag).")

    elif args.input_subfolder:
        input_base_path = args.input_subfolder # This is directly the patient folder
        if not input_base_path.is_dir():
            logger.error(f"Error: The provided single subfolder '{input_base_path}' is not a valid directory.")
            exit(1)
        patient_folders_to_process.append(input_base_path)
        use_multiprocessing = False # Always false for single subfolder
        logger.info(f"Processing single patient subfolder: {input_base_path.name}")

    if not patient_folders_to_process:
        logger.warning("No patient folders found to process. Exiting.")
        exit(0)

    start_time = time.time()
    if use_multiprocessing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_single_patient_folder, case, args.output_path, args, args.log)
                for case in patient_folders_to_process
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Error during processing: {e}")    
    else:
        for case in patient_folders_to_process:
            try:
                process_single_patient_folder(case, args.output_path, args, args.log)
                logger.info(f"Finished processing patient: {case.name}")
            except Exception as e:
                logger.error(f"An error occurred while processing patient {case.name}: {e}")

    logger.info(f"Total cases processed: {len(patient_folders_to_process)}")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
    logger.success(f'Done. Results are saved in {args.output_path}')