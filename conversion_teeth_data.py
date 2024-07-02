import multiprocessing as mp

import click
import nibabel as nib
import numpy as np
from pathlib2 import Path

from dataset import teeth


@click.command()
@click.option('-d', '--data', help='teeth data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True), default='teeth_data_good', required=True)
@click.option('-o', '--output', help='output npy file path',
              type=click.Path(dir_okay=True, resolve_path=True), default='data_good',required=True)
def conversion_all(data, output):
    data = Path(data)
    output = Path(output)

    cases = sorted([d for d in data.iterdir() if d.is_dir()])
    pool = mp.Pool()
    pool.map(conversion, zip(cases, [output] * len(cases)))
    pool.close()
    pool.join()


def conversion(data):
    case, output = data
    vol_nii = nib.load(str(case / (case.name + '_teeth.nii.gz')))
    print(str(case / (case.name + '_teeth.nii.gz')))
    vol = vol_nii.get_data()
    vol = teeth.normalize(vol)

    print(vol.shape)

    imaging_dir = output / case.name / 'teeth_img'
    if not imaging_dir.exists():
        imaging_dir.mkdir(parents=True)
    if len(list(imaging_dir.glob('*.npy'))) != vol.shape[2]:
        for i in range(vol.shape[2]):
            np.save(str(imaging_dir / f'{i:03}.npy'), vol[:,:,i])

    segmentation_file = case / ('prediction_' + str(case.name) + '.nii.gz')
    if segmentation_file.exists():
        seg = nib.load(str(case / ('prediction_' + str(case.name) + '.nii.gz'))).get_data()
        print(('prediction_' + str(case.name) + '.nii.gz'))
        segmentation_dir = output / case.name / 'segmentation'
        if not segmentation_dir.exists():
            segmentation_dir.mkdir(parents=True)
        if len(list(segmentation_dir.glob('*.npy'))) != seg.shape[2]:
            for i in range(seg.shape[2]):
                np.save(str(segmentation_dir / f'{i:03}.npy'), seg[:,:,i])

    affine_dir = output / case.name
    if not affine_dir.exists():
        affine_dir.mkdir(parents=True)
    affine = vol_nii.affine
    np.save(str(affine_dir / 'affine.npy'), affine)


if __name__ == '__main__':
    conversion_all()
