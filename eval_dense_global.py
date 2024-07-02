import click

import nibabel as nib
import numpy as np
import torch

from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import Teeth
from dataset.teethTransform import TeethTransform
from network import MD2UNet
from utils.vis import imshow


@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(640, 640), show_default=True)
@click.option('-d', '--data', 'data_path', help='Path of teeth data after conversion',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='/home/wr/MD2UNet/data_good/', show_default=True)
@click.option('-r', '--resume', help='Resume model',
              type=click.Path(exists=True, file_okay=True, resolve_path=True),
              default="/home/wr/MD2UNet/runs/4-4test2024-04-10_20-01-27/checkpoint/cp_001.pth", required=True)
@click.option('-o', '--output', 'output_path', help='output image path',
              type=click.Path(dir_okay=True, resolve_path=True), default='/home/wr/MD2UNet/out_proc', show_default=True)
@click.option('--vis_intvl', help='Number of iteration interval of display visualize image. '
                                  'No display when set to 0',
              type=int, default=1, show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def main(batch_size, num_gpu, img_size, data_path, resume, output_path, vis_intvl, num_workers):
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)


    transform = TeethTransform(output_size=img_size,use_roi=True)

    dataset = Teeth(data_path, stack_num=3, spec_classes=[0, 1, 2], img_size=img_size, use_roi=True, 
                    roi_file='roi.json', test_transform=transform)

    net = MD2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)

    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')

    gpu_ids = [i for i in range(num_gpu)]

    print(f'{" Start evaluation ":-^40s}\n')
    msg = f'Net: {net.__class__.__name__}\n' + \
          f'Dataset: {dataset.__class__.__name__}\n' + \
          f'Batch size: {batch_size}\n' + \
          f'Device: cuda{str(gpu_ids)}\n'
    print(msg)

    torch.cuda.empty_cache()

    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()

    net.eval()
    torch.set_grad_enabled(False)
    transform.eval()

    subset = dataset.test_dataset
    case_slice_indices = dataset.test_case_slice_indices

    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)

    case = 0
    vol_output = []

    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/test', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, idx = data['image'].cuda(), data['index']  

            outputs = net(imgs)

            predicts = outputs['output']
            predicts = predicts.argmax(dim=1)

            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()

            vol_output.append(predicts)
           
            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]
                vol_ = vol_output[:vol_num_slice]
                vol_ = vol_.astype(np.uint8)
                vol_ = vol_.transpose(1, 2, 0)
                case_id = dataset.case_idx_to_case_id(case, type='test')
                affine = np.load(data_path / f'{case_id}' / 'affine.npy')
                vol_nii = nib.Nifti1Image(vol_, affine)
                vol_nii_filename = output_path / f'prediction_{case_id}.nii.gz'
                
                print(vol_nii_filename)
                nib.save(vol_nii,str(vol_nii_filename))
                

                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)

            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                data['predict'] = predicts
                data = dataset.vis_transform(data)
                imgs, predicts = data['image'], data['predict']
                imshow(title=f'eval/test', imgs=(imgs[0, 1], predicts[0]), shape=(1, 2),
                       subtitle=('image', 'predict'))



if __name__ == '__main__':
    main()
