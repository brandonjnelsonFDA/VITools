'''
module for working with phantoms
'''

from pathlib import Path

import numpy as np
from monai.transforms import Resize, RandAffine, Affine

from . import dicom_to_voxelized_phantom
from .artifact_generation import transform_image_label_pair


def resize(phantom, shape):
    resize = Resize(max(shape), size_mode='longest')
    resized = resize(phantom[None])[0]
    return resized


def voxelize_ground_truth(dicom_path: str | Path, phantom_path: str | Path,
                          material_threshold_dict: dict | None = None):
    '''
    Used to convert ground truth image into segmented volumes used by XCIST to
    run simulations

    :param dicom_path: str | Path, path where the DICOM images are located,
        these are typically the output of `convert_to_dicom`
    :param phantom_path: str or Path, where the phantom files are to be
        written
    :param material_threshold_dict: dictionary mapping XCIST materials to
        appropriate lower thresholds in the ground truth image, see the .cfg
        here for examples
        <https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized>
    '''
    nfiles = len(list(Path(dicom_path).rglob('*.dcm')))
    slice_range = list(range(nfiles))
    if not material_threshold_dict:
        material_threshold_dict = dict(zip(
                                        ['ncat_adipose',
                                         'ncat_water',
                                         'ncat_brain',
                                         'ncat_skull'],
                                        [-200, -10, 10, 300]))

    cfg_file_str = f"""
# Path where the DICOM images are located:
phantom.dicom_path = '{dicom_path}'
# Path where the phantom files are to be written
# (the last folder name will be the phantom files' base name):
phantom.phantom_path = '{phantom_path}'
phantom.materials = {list(material_threshold_dict.keys())}
phantom.mu_energy = 60
phantom.thresholds = {list(material_threshold_dict.values())}
phantom.slice_range = [{[slice_range[0], slice_range[-1]]}] # Range of DICOM
# image numbers to include. (first, last slice)
phantom.show_phantom = False  # Flag to turn on/off image display.
phantom.overwrite = True  # Flag to overwrite existing files without warning.
"""

    dicom_to_voxel_cfg = phantom_path / 'dicom_to_voxelized.cfg'

    with open(dicom_to_voxel_cfg, 'w') as f:
        f.write(cfg_file_str)

    dicom_to_voxelized_phantom.run_from_config(dicom_to_voxel_cfg)


class Phantom:
    '''
    Base phantom that can accept any img array and spacings which
        specify the size

    :param img: 2D or 3D numpy array defining the phantom
    :param spacings: tuple, voxel spacings [mm] (z, x, y), defining voxel
        sizes, defaults to 1 mm in each direction
    :param patient_name: patient identifier to be saved in DICOM header
    :param patientid: int, patient identifier to be saved in DICOM header
    :param age: float, in years to be saved in DICOM header
    '''
    def __init__(self, img: np.ndarray, spacings: tuple = (1, 1, 1),
                 patient_name='default', patientid=0, age=0) -> None:
        self._phantom = img
        self.dz, self.dx, self.dy = spacings
        self.patient_name = patient_name
        self.patientid = patientid
        self.age = age

    def __repr__(self) -> str:
        repr = f'''
        phantom class: {self.__class__.__name__}
        age [yrs]: {self.age}
        shape [voxels]: {self.shape}
        size [mm]: {self.size}
        '''
        return repr

    def get_CT_number_phantom(self) -> np.ndarray:
        return self._phantom

    @property
    def spacings(self):
        return self.dz, self.dx, self.dy

    @property
    def shape(self):
        return list(self._phantom.shape)

    @property
    def size(self):
        return np.array(self.spacings)*self.shape
