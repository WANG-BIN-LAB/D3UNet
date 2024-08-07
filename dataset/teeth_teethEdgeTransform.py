import cv2
import numpy as np
from albumentations import Compose as Compose_albu
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    GridDistortion,
    RandomBrightnessContrast,
    RandomGamma,
    Crop,
    LongestMaxSize,
    ShiftScaleRotate
)


def to_numpy(data):

    image, label, teeth_edge, root_edge = data['image'], data['label'], data['teeth_edge'], data['root_edge']

    data['image'] = np.array(image)
    if data['label'] is not None:
        data['label'] = np.array(label)

    if data['teeth_edge'] is not None:
        data['teeth_edge'] = np.array(teeth_edge)

    if data['root_edge'] is not None:
        data['root_edge'] = np.array(root_edge)

    return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class TeethEdgeTransform:
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def __call__(self, data):
        data = to_numpy(data)
        img, label, teeth_edge, root_edge = data['image'], data['label'], data['teeth_edge'], data['root_edge']
        is_3d = True if img.shape == 4 else False  

        max_size = max(self._output_size[0], self._output_size[1])

        if self._type == 'train':
            task = [
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5),
                GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.5),
                LongestMaxSize(max_size, p=1),
                PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1),

            ]
        else:
            task = [
                LongestMaxSize(max_size, p=1),
                PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1)
            ]

        if self.use_roi:
            assert 'roi' in data.keys() and len(data['roi']) is not 0
            roi = data['roi']
            min_y = 0
            max_y = img.shape[0]
            min_z = 0
            max_z = img.shape[1]
            min_z = max(min_z, roi['min_z'] - self._roi_error_range)
            max_z = min(max_z, roi['max_z'] + self._roi_error_range)
            min_y = max(min_y, roi['min_y'] - self._roi_error_range)
            max_y = min(max_y, roi['max_y'] + self._roi_error_range)

            crop = [Crop(min_z, min_y, max_z, max_y, p=1)]
#            task = crop + task
            task_roi = [Crop(min_z, min_y, max_z, max_y, p=1)]
        else:
            task_roi = []

        aug_roi = Compose_albu(task_roi)
        aug = Compose_albu(task)
        if not is_3d:
            # aug_data = aug(image=img, mask=label)
            aug_data = aug_roi(image=img, masks=[label, teeth_edge, root_edge])
            aug_data = aug(image=img, masks=[label, teeth_edge, root_edge])
            data['image'], data['label'], data['teeth_edge'], data['root_edge'] = aug_data['image'], aug_data['masks'][0], aug_data['masks'][1], aug_data['masks'][2]

        # 3D aug function BUG !!!
        else:
            keys = {}
            targets = {}

            for i in range(1, img.shape[2]):
                keys.update({f'image{i}': 'image'})
                keys.update({f'mask{i}': 'mask'})
                keys.update({f'mask_teeth_edge{i}': 'mask_teeth_edge'})
                targets.update({f'image{i}': img[:, :, i]})
                targets.update({f'mask{i}': label[:, :, i]})
                targets.update({f'mask_teeth_edge{i}': teeth_edge[:, :, i]})
            aug.add_targets(keys)

            targets.update({'image': img[:, :, 0]})
            targets.update({'mask': label[:, :, 0]})
            targets.update({'mask_teeth_edge': teeth_edge[:, :, 0]})

            aug_data = aug(**targets)
            imgs = [aug_data['image']]
            labels = [aug_data['mask']]
            teeth_edges = [aug_data['mask_teeth_edge']]

            for i in range(1, img.shape[2]):
                imgs.append(aug_data[f'image{i}'])
                labels.append(aug_data[f'mask{i}'])
                teeth_edges.append(aug_data[f'mask_teeth_edge{i}'])

            img = np.stack(imgs, axis=-1)
            label = np.stack(labels, axis=-1)
            teeth_edge = np.stack(teeth_edges, axis=-1)
            data['image'] = img
            data['label'] = label
            data['teeth_edge'] = teeth_edge

        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size
