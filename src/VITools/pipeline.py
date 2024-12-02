'''
pipeline: this high level module organizes the healthy head phantoms,
lesion definitions, augmentations, and CT simulation together into the final
ct_simulation function.
'''
from pathlib import Path
from shutil import rmtree

import numpy as np
import pydicom
import pandas as pd
from scipy.ndimage import center_of_mass

from .image_acquisition import Scanner, read_dicom


def load_vol(file_list):
    return np.stack(list(map(read_dicom, file_list)))


class Study:
    def __init__(self, scanner: Scanner, study_name='default'):
        self.scanner = scanner
        self.phantom = scanner.phantom
        self.study_name = study_name
        self.metadata = None

    def __repr__(self) -> str:
        repr = f'''
        study name: {self.study_name}
        Phantom details:
        ----------------
        {self.scanner.phantom.__repr__()}

        Scanner details:
        ----------------
        Scanner: {self.scanner.__repr__()}

        Study details:
        --------------
        {self.metadata}
        '''
        return repr

    @property
    def shape(self):
        return list(self.phantom._phantom.shape)

    @property
    def size(self):
        return np.array(self.phantom.spacings)*self.phantom._phantom.shape

    def run_study(self, output_directory=None, kVp=120, mA=200, views=1000,
                  fov=250, zspan='dynamic',
                  kernel='standard', slice_thickness=1, **kwargs):
        patient_name = self.phantom.patient_name
        age = self.phantom.age
        lesion_type = self.phantom.lesion_type
        intensity = self.phantom.lesion_intensity

        ct = self.scanner
        if zspan == 'dynamic':
            startZ, endZ = ct.recommend_scan_range()
        elif isinstance(zspan, tuple | list):
            startZ, endZ = zspan

        ct.run_scan(startZ=startZ, endZ=endZ, views=views,
                    mA=mA, kVp=kVp)
        ct.run_recon(fov=fov, kernel=kernel, sliceThickness=slice_thickness)
        self.scanner = ct
        self.images = ct.recon
        if output_directory is None:
            output_directory = self.scanner.output_dir
        else:
            output_directory = Path(output_directory) / patient_name
        dicom_path = output_directory / 'dicoms'
        dcm_files = ct.write_to_dicom(dicom_path / f'{patient_name}.dcm')

        mask_files = [None]*len(dcm_files)
        z, x, y = 3*[None]
        vol_by_slice_mL = [0]*len(dcm_files)
        vol_ml = 0
        if lesion_type:
            lesion_only = ct
            mask = ct.get_lesion_mask(startZ=startZ, endZ=endZ,
                                      slice_thickness=slice_thickness, fov=fov)

            lesion_only.recon = mask
            dicom_path = output_directory / 'lesion_masks'
            mask_files = lesion_only.write_to_dicom(dicom_path /
                                                    f'{patient_name}_mask.dcm')
            mask = load_vol(mask_files)
            self.lesion = mask & (self.images > self.images.mean())
            self.scanner.recon = self.images

            dcm = pydicom.read_file(mask_files[0])
            spacings = list(map(float, [dcm.SliceThickness] +
                            list(dcm.PixelSpacing)))

            vol_ml = np.prod(spacings) * mask.sum() / 1000
            vol_by_slice_mL = np.prod(spacings) *\
                self.lesion.sum(axis=(1, 2)) / 1000
            z, x, y = center_of_mass(mask)
            self.lesion_coords = (z, x, y)
        ages = []
        names = []
        files = []
        kVps = []
        mA_list = []
        fovs = []
        kernels = []
        views_list = []
        masks = []
        intensity_list = []
        lesion_type_list = []
        mass_effect = []
        center_x_list = []
        center_y_list = []
        center_z_list = []
        lesion_volume_list = []

        for f, m, vol_ml in zip(dcm_files, mask_files, vol_by_slice_mL):
            names.append(patient_name)
            ages.append(age)
            files.append(f)
            kVps.append(kVp)
            mA_list.append(mA)
            fovs.append(fov)
            kernels.append(kernel)
            views_list.append(views)
            masks.append(m)

            if vol_ml > 0:
                slice_mass_effect = self.phantom.mass_effect
                slice_intensity = intensity
                slice_x = int(x)
                slice_y = int(y)
                slice_z = int(z)
                slice_type = lesion_type
            else:
                slice_mass_effect = None
                slice_intensity = None
                slice_x = None
                slice_y = None
                slice_z = None
                slice_type = None

            intensity_list.append(slice_intensity)
            lesion_type_list.append(slice_type)
            mass_effect.append(slice_mass_effect)
            center_x_list.append(slice_x)
            center_y_list.append(slice_y)
            center_z_list.append(slice_z)
            lesion_volume_list.append(vol_ml)

        metadata = pd.DataFrame({'name': names,
                                 'age': ages,
                                 'intensity': intensity_list,
                                 'center x': center_x_list,
                                 'center y': center_y_list,
                                 'center z': center_z_list,
                                 'lesion type': lesion_type_list,
                                 'mass effect': mass_effect,
                                 'lesion volume [mL]': lesion_volume_list,
                                 'mA': mA_list,
                                 'kVp': kVps,
                                 'views': views_list,
                                 'fov [mm]': fovs,
                                 'kernel': kernels,
                                 'image file': files,
                                 'mask file': masks})
        self.metadata = metadata
        return self
