from pathlib import Path

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact


# https://radiopaedia.org/articles/windowing-ct?lang=us
display_settings = {
    'brain': (80, 40),
    'subdural': (300, 100),
    'stroke': (40, 40),
    'temporal bones': (2800, 600),
    'soft tissues': (400, 50),
    'lung': (1500, -600),
    'liver': (150, 30),
}


def ctshow(img, window='soft tissues'):
    # Define some specific window settings here
    if isinstance(window, str):
        if window not in display_settings:
            raise ValueError(f"{window} not in {display_settings}")
        ww, wl = display_settings[window]
    elif isinstance(window, tuple):
        ww = window[0]
        wl = window[1]
    else:
        ww = 6.0 * img.std()
        wl = img.mean()

    if img.ndim == 3:
        img = img[0].copy()

    plt.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)
    plt.xticks([])
    plt.yticks([])
    return


def circle_select(img, xy, r):
    assert img.ndim == 2
    circle_mask = np.zeros_like(img)
    for i in range(circle_mask.shape[0]):
        for j in range(circle_mask.shape[1]):
            if (i-xy[0])**2 + (j-xy[1])**2 < r**2:
                circle_mask[i, j] = True
    return circle_mask.astype(bool)


def browse_studies(metadata, phantom='CTP404', fov=12.3, dose=100, recon='fbp', kernel='D45', repeat=0, display='soft tissues'):
    phantom_df = metadata[(metadata.phantom == phantom) &
                          (metadata.recon == recon)]
    available_fovs = sorted(phantom_df['FOV [cm]'].unique())
    if fov not in available_fovs:
        print(f'FOV {fov} not in {available_fovs}')
        return
    patient = phantom_df[phantom_df['FOV [cm]'] == fov]
    if (recon != 'ground truth') and (recon != 'noise free'):
        available_doses = sorted(patient['Dose [%]'].unique())
        if dose not in available_doses:
            print(f'dose {dose}% not in {available_doses}')
            return
        patient = patient[(patient['Dose [%]'] == dose) &
                          (patient['kernel'] == kernel) &
                          (patient['repeat'] == repeat)]
    dcm_file = patient.file.item()
    dcm = pydicom.dcmread(dcm_file)
    img = dcm.pixel_array + int(dcm.RescaleIntercept)

    ww, wl = display_settings[display]
    minn = wl - ww/2
    maxx = wl + ww/2
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=minn, vmax=maxx)
    plt.colorbar(label=f'HU | {display} [ww: {ww}, wl: {wl}]')
    plt.title(patient['Name'].item())


def study_viewer(metadata):
    viewer = lambda **kwargs: browse_studies(metadata, **kwargs)
    interact(viewer,
             phantom=metadata.phantom.unique(),
             dose=sorted(metadata['Dose [%]'].unique(), reverse=True),
             fov=sorted(metadata['FOV [cm]'].unique()),
             recon=metadata['recon'].unique(),
             kernel=metadata['kernel'].unique(),
             repeat=metadata['repeat'].unique(),
             display=display_settings.keys())
