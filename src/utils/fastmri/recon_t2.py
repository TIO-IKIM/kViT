# -*- coding: utf-8 -*-
import numpy as np
import h5py
import torch
import sys
from glob import glob

sys.path.append("/home/jovyan/k-radiomics-storage/kspace-radiomics/src/utils")
sys.path.append("/home/jovyan/k-radiomics-storage/kspace-radiomics/src/utils/fastmri")
sys.path.append("/home/jovyan/k-radiomics-storage/k-radiomics/reconstruction/tools")

import xml.etree.ElementTree as etree
from grappa import Grappa
import sigpy as sp
import sigpy.mri as mri
import argparse
from tools import espirit_combine
import logging

parser = argparse.ArgumentParser(
    prog="Reconstruction",
)
parser.add_argument(
    "-i",
    type=str,
    help="Path to input folder",
    default="/home/jovyan/radiology/fastmri_prostate/fastMRI_prostate_T2_IDS_001_020",
)


def zero_pad_kspace_hdr(hdr: str, unpadded_kspace: np.ndarray) -> np.ndarray:
    """
    Perform zero-padding on k-space data to have the same number of
    points in the x- and y-directions.

    Parameters
    ----------
    hdr : str
        The XML header string.
    unpadded_kspace : array-like of shape (sl, ro , coils, pe)
        The k-space data to be padded.

    Returns
    -------
    padded_kspace : ndarray of shape (sl, ro_padded, coils, pe_padded)
        The zero-padded k-space data, where ro_padded and pe_padded are
        the dimensions of the readout and phase-encoding directions after
        padding.

    Notes
    -----
    The padding value is calculated using the `get_padding` function, which
    extracts the padding value from the XML header string. If the difference
    between the readout dimension and the maximum phase-encoding dimension
    is not divisible by 2, the padding is applied asymmetrically, with one
    side having an additional zero-padding.

    """
    padding = get_padding(hdr)
    if padding % 2 != 0:
        padding_left = int(np.floor(padding))
        padding_right = int(np.ceil(padding))
    else:
        padding_left = int(padding)
        padding_right = int(padding)
    padded_kspace = np.pad(
        unpadded_kspace, ((0, 0), (0, 0), (0, 0), (padding_left, padding_right))
    )

    return padded_kspace


def get_padding(hdr: str) -> float:
    """
    Extract the padding value from an XML header string.

    Parameters:
    -----------
    hdr : str
        The XML header string.

    Returns:
    --------
    float
        The padding value calculated as (x - max_enc)/2, where x is the readout dimension and
        max_enc is the maximum phase-encoding dimension.
    """
    et_root = etree.fromstring(hdr)
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
    enc = ["encoding", "encodedSpace", "matrixSize"]
    enc_x = int(et_query(et_root, enc + ["x"]))
    padding = (enc_x - enc_limits_max) / 2

    return padding


def et_query(
    root: etree.Element, qlist, namespace: str = "http://www.ismrm.org/ISMRMRD"
) -> str:
    """
    ElementTree query function.

    This function queries an XML document using ElementTree.

    Parameters:
    -----------
    root : Element
        Root of the XML document to search through.
    qlist : Sequence of str
        A sequence of strings for nested searches, e.g., ["Encoding", "matrixSize"].
    namespace : str, optional
        XML namespace to prepend query.

    Returns:
    --------
    str
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def create_coil_combined_im(
    multicoil_multislice_kspace: np.ndarray, device
) -> np.ndarray:
    """
    Create a coil combined image from a multicoil-multislice k-space array.

    Parameters:
    -----------
    multicoil_multislice_kspace : array-like
        Input k-space data with shape (slices, coils, readout, phase encode).

    Returns:
    --------
    image_mat : array-like
        Coil combined image data with shape (slices, x, y).
    """

    k = multicoil_multislice_kspace
    image_mat = np.zeros((k.shape[0], k.shape[2], k.shape[3]), dtype=complex)
    for i in range(image_mat.shape[0]):
        data_sl = k[i, :, :, :]
        mps = mri.app.EspiritCalib(
            data_sl, 32, 0.005, show_pbar=False, device=device
        ).run()
        image_mat[i, :, :] = espirit_combine(data_sl, mps)
    return image_mat


def t2_reconstruction(kspace, hdr, calib_data):
    device = sp.Device(0)  # Use GPU 0

    num_avg, num_slices, num_coils, num_ro, num_pe = kspace.shape

    grappa_weight_dict = {}
    grappa_weight_dict_2 = {}

    kspace_slice_regridded = kspace[0, 0, ...]
    grappa_obj = Grappa(
        np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
    )
    kspace_slice_regridded_2 = kspace[1, 0, ...]
    grappa_obj_2 = Grappa(
        np.transpose(kspace_slice_regridded_2, (2, 0, 1)),
        kernel_size=(5, 5),
        coil_axis=1,
    )

    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        calibration_regridded = calib_data[slice_num, ...]
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )
        grappa_weight_dict_2[slice_num] = grappa_obj_2.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace.shape, dtype=complex)

    for average, grappa_obj, grappa_weight_dict in zip(
        [0, 1, 2],
        [grappa_obj, grappa_obj_2, grappa_obj],
        [grappa_weight_dict, grappa_weight_dict_2, grappa_weight_dict],
    ):
        for slice_num in range(num_slices):
            kspace_slice_regridded = kspace[average, slice_num, ...]
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dict[slice_num],
            )
            kspace_post_grappa_all[average, slice_num, ...] = np.moveaxis(
                np.moveaxis(kspace_post_grappa, 0, 1), 1, 2
            )

    # recon image for each average
    im = np.zeros((num_avg, num_slices, num_ro, num_ro), dtype=complex)
    for average in range(num_avg):
        kspace_grappa = kspace_post_grappa_all[average, ...]
        kspace_grappa_padded = zero_pad_kspace_hdr(hdr, kspace_grappa)
        im[average, ...] = create_coil_combined_im(kspace_grappa_padded, device)

    im = np.mean(im, axis=0)

    torch.save(torch.tensor(im), file.replace(".h5", "_recon.pt"))


if __name__ == "__main__":
    args = parser.parse_args()

    folder_path = args.i
    logging.basicConfig(
        filename=f"{folder_path}/recon.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    file_list = glob(folder_path + "/*.h5")

    for file in file_list:
        try:
            logging.info(f"Processing file: {file}")
            f = h5py.File(file, "r")
            kspace = f["kspace"][:]
            calibration = f["calibration_data"][:]
            hdr = f["ismrmrd_header"][()]
            t2_reconstruction(kspace, hdr, calibration)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            continue
