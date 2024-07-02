import sys
import datetime
import time
import click
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from pathlib2 import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from scipy import ndimage as ndi
import utils.checkpoint as cp
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import dilation, erosion, square
from skimage.util import img_as_float, view_as_windows
from skimage.color import gray2rgb
from dataset import Teeth
from dataset import TeethEdge
from dataset.transform import MedicalTransform
from dataset.teethTransform import TeethTransform
from dataset.teeth_teethEdgeTransform import TeethEdgeTransform
from loss import GeneralizedDiceLoss
from loss.util import class2one_hot
from network import MD2UNet
from utils.metrics import Evaluator
from utils.vis import imshow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@click.command()
@click.option('-e', '--epoch', 'epoch_num', help='Number of training epoch', type=int, default=30, show_default=True)
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=8, show_default=True)
@click.option('-l', '--lr', help='Learning rate', type=float, default=0.0001, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(256, 256), show_default=True)
@click.option('-d', '--data', 'data_path', help='Path of teeth data after conversion',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='/home/wr/MD2UNet/data_good/', show_default=True)
@click.option('--log', 'log_path', help='Checkpoint and log file save path',
              type=click.Path(dir_okay=True, resolve_path=True),
              default='/home/wr/MD2UNet/runs/4-4test', show_default=True)
@click.option('-r', '--resume', help='Resume checkpoint file to continue training',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), default=None)
@click.option('--eval_intvl', help='Number of epoch interval of evaluation. '
                                   'No evaluation when set to 0',
              type=int, default=1, show_default=True)
@click.option('--cp_intvl', help='Number of epoch interval of checkpoint save. '
                                 'No checkpoint save when set to 0',
              type=int, default=1, show_default=True)
@click.option('--vis_intvl', help='Number of iteration interval of display visualize image. '
                                  'No display when set to 0',
              type=int, default=2, show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=4, show_default=True)
def main(epoch_num, batch_size, lr, num_gpu, img_size, data_path, log_path,
         resume, eval_intvl, cp_intvl, vis_intvl, num_workers):
    data_path = Path(data_path)

    now = time.time()
    timeArray = time.localtime(now)
    now = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
    log_path = Path(log_path + str(now))

    cp_path = log_path / 'checkpoint'
    
    if not resume and log_path.exists() and len(list(log_path.glob('*'))) > 0:

        print(f'log path "{str(log_path)}" has old file', file=sys.stderr)
        sys.exit(-1)
    if not cp_path.exists():
        cp_path.mkdir(parents=True)
    
    transform = TeethEdgeTransform(output_size=img_size, roi_error_range=15, use_roi=False)
    
    dataset = TeethEdge(data_path, stack_num=3, spec_classes=[0, 1, 2], img_size=img_size,
                     use_roi=False, roi_file='roi.json', roi_error_range=5,
                     train_transform=transform, valid_transform=transform)
    
    net = MD2UNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=0.0005)
    
    start_epoch = 0
    if resume:
        data = {
            'net': net,
            'optimizer': optimizer,
            'epoch': 0
        }
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')
        start_epoch = data['epoch'] + 1
    
    criterion = GeneralizedDiceLoss(idc=[0, 1, 2])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )
    
    logger = SummaryWriter(str(log_path))
    
    gpu_ids = [i for i in range(num_gpu)]
    
    print(f'{" Start training ":-^40s}\n')
    msg = f'Net: {net.__class__.__name__}\n' + \
          f'Dataset: {dataset.__class__.__name__}\n' + \
          f'Epochs: {epoch_num}\n' + \
          f'Learning rate: {optimizer.param_groups[0]["lr"]}\n' + \
          f'Batch size: {batch_size}\n' + \
          f'Device: cuda{str(gpu_ids)}\n'
    print(msg)
    
    torch.cuda.empty_cache()
    
    # to GPU device
    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
    criterion = criterion.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    
    # start training
    valid_score = 0.0
    best_score = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, epoch_num):
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        
        net.train()
        torch.set_grad_enabled(True)
        transform.train()
        try:
            loss = training(net, dataset, criterion, optimizer, scheduler,
                            epoch, batch_size, num_workers, vis_intvl, logger)
            
            if eval_intvl > 0 and (epoch + 1) % eval_intvl == 0:
                net.eval()
                torch.set_grad_enabled(False)
                transform.eval()
                
                train_score = evaluation(net, dataset, epoch, batch_size, num_workers, vis_intvl, logger, type='train')
                valid_score = evaluation(net, dataset, epoch, batch_size, num_workers, vis_intvl, logger, type='valid')
                
                print(f'Train data score: {train_score:.5f}')
                print(f'Valid data score: {valid_score:.5f}')
            
            if valid_score > best_score:
                best_score = valid_score
                best_epoch = epoch
                cp_file = cp_path / 'best.pth'
                cp.save(epoch, net.module, optimizer, str(cp_file))
                print('Update best acc!')

                logger.add_scalar('best/epoch', best_epoch + 1, 0)
                logger.add_scalar('best/score', best_score, 0)
            
            if (epoch + 1) % cp_intvl == 0:
                cp_file = cp_path / f'cp_{epoch + 1:03d}.pth'
                cp.save(epoch, net.module, optimizer, str(cp_file))
            
            print(f'Best epoch: {best_epoch + 1}')
            print(f'Best score: {best_score:.5f}')
        
        except KeyboardInterrupt:
            cp_file = cp_path / 'INTERRUPTED.pth'
            cp.save(epoch, net.module, optimizer, str(cp_file))
            return


def training(net, dataset, criterion, optimizer, scheduler, epoch, batch_size, num_workers, vis_intvl, logger):
    sampler = RandomSampler(dataset.train_dataset)
    
    train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    
    tbar = tqdm(train_loader, ascii=True, desc='train', dynamic_ncols=True)
    for batch_idx, data in enumerate(tbar):
        # imgs, labels = data['image'].cuda(), data['label'].cuda()

        #imgs, labels, teeth_edges = data['image'].cuda(), data['label'].cuda(), data['teeth_edge'].cuda()
        imgs, labels, teeth_edges, root_edges = data['image'].cuda(), data['label'].cuda(), data['teeth_edge'].cuda(), data['root_edge'].cuda()


        outputs = net(imgs)

        losses = {}
        for key, up_outputs in outputs.items():
            b, c, h, w = up_outputs.shape
            up_labels = torch.unsqueeze(labels.float(), dim=1)
            up_labels = F.interpolate(up_labels, size=(h, w), mode='bilinear')
            up_labels = torch.squeeze(up_labels, dim=1).long()
            up_labels_onehot = class2one_hot(up_labels, 3)
            # edge loss cal

            # up_teeth_edge_labels = torch.unsqueeze(teeth_edges.float(), dim=1)
            # up_teeth_edge_labels = F.interpolate(up_teeth_edge_labels, size=(h, w), mode='bilinear')
            # up_teeth_edge_labels = torch.squeeze(up_teeth_edge_labels, dim=1).long()
            # up_teeth_edge_labels_onehot = class2one_hot(up_teeth_edge_labels, 3)

            up_outputs = F.softmax(up_outputs, dim=1)
            #up_outputs = F.log_softmax(up_outputs, dim=1)
            up_loss = criterion(up_outputs, up_labels_onehot)

            # up_teeth_edge_loss = criterion(up_outputs, up_teeth_edge_labels_onehot)

            #up_loss_sum = up_loss + up_teeth_edge_loss

            #print('loss' + ':' + str(up_loss) + '  EdgeLoss' + " : " + str(up_teeth_edge_loss) + '  loss_sum' + " : " + str(up_loss_sum))
            #losses[key] = up_loss

            # losses[key] = up_loss_sum
            losses[key] = up_loss

        predicts = outputs['output']#[8,3,256,256]
        predicts = predicts.argmax(dim=1)#[8,256,256]
        predicts = predicts.cpu().detach().numpy()
        predicts_root_area = np.where(predicts > 1, predicts, 0)#[8,256,256] 像素点值不>1的替换为0(即只有值为2的像素点被保留，root)
        predicts_teeth_area = predicts - predicts_root_area #牙齿teeth[8,256,256]
        predicts_boundary_teeth = np.zeros(predicts_teeth_area.shape)#[8,256,256]
        predicts_boundary_root = np.zeros(predicts_root_area.shape)#[8,256,256]
        # print(f"Size of predicts_boundary_teeth: {predicts_boundary_teeth.shape}")
        # print(f"Size of predicts_teeth_area: {predicts_teeth_area.shape}")
        #for i in range(batch_size):
        for i in range(len(predicts_teeth_area)):
            if predicts_teeth_area[i] is not None:
                predicts_boundary_teeth[i] = find_boundaries(predicts_teeth_area[i], mode='inner').astype(np.int16)
            if predicts_root_area[i] is not None:
                predicts_boundary_root[i] = find_boundaries(predicts_root_area[i], mode='inner').astype(np.int16)

        # boundary_root = find_boundaries(root_area, mode='inner').astype(np.int16)

        teeth_boundary_label = teeth_edges.cpu().detach().numpy()
        root_boundary_label = root_edges.cpu().detach().numpy()

        dice_teeth_edge_loss = dice_edge_loss(teeth_boundary_label, predicts_boundary_teeth)
        dice_root_edge_loss = dice_edge_loss(root_boundary_label, predicts_boundary_root)

        #print('/n')
        #print('no_edge_loss:' + str(sum(losses.values()).item()) + '  teeth_edge_loss:' + str(dice_teeth_edge_loss) + ' root_edge_loss:' + str(dice_root_edge_loss))

        loss = sum(losses.values()) 
#        loss = sum(losses.values()) +  0.1*dice_teeth_edge_loss + 0.1*dice_root_edge_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if vis_intvl > 0 and batch_idx % vis_intvl == 0:
            data['predict'] = outputs['output']
            data = dataset.vis_transform(data)
            imgs, labels, teeth_edges, predicts_vis,root_edges = data['image'], data['label'], data['teeth_edge'],data['predict'],data['root_edge']
            # imgs, labels,predicts = data['image'], data['label'], data['predict']

            # imshow(title='Train', imgs=(imgs[0, dataset.img_channels // 2], labels[0], predicts[0]),
            #        shape=(1, 3), subtitle=('image', 'label', 'predict'))

            # imshow(title='Train', imgs=(imgs[0, dataset.img_channels // 2], labels[0], teeth_edges[0], predicts[0]),
            #        shape=(1, 4), subtitle=('image', 'label', 'teeth_edge', 'predict'))

            # predicts_root_area_vis = np.where(predicts_vis > 1.0, predicts_vis, 0)
            # predicts_teeth_area_vis = predicts_vis - predicts_root_area_vis
            # predicts_boundary_teeth_vis = np.zeros(predicts_teeth_area_vis.shape)
            # for i in range(batch_size):
            #     predicts_boundary_teeth_vis[i][0] = find_boundaries(predicts_teeth_area_vis[i][0], mode='inner').astype(np.float32)
            predicts_root_area_vis = np.where(predicts_vis > 1.0, predicts_vis, 0)
            predicts_teeth_area_vis = predicts_vis - predicts_root_area_vis
            predicts_boundary_teeth_vis = np.zeros(predicts_teeth_area_vis.shape)
            predicts_boundary_root_vis = np.zeros(predicts_root_area_vis.shape)


            for i in range(len(predicts_teeth_area_vis)):
                predicts_boundary_teeth_vis[i][0] = find_boundaries(predicts_teeth_area_vis[i][0], mode='inner').astype(
                    np.float32)
                predicts_boundary_root_vis[i][0] = find_boundaries(predicts_vis[i][2], mode='inner').astype(np.float32)



            # imshow(title='Train', imgs=(imgs[0, dataset.img_channels // 2], labels[0], predicts_vis[0], teeth_edges[0], predicts_boundary_teeth_vis[0]),
            #        shape=(1, 5), subtitle=('image', 'label', 'predict', 'teeth_edge', 'edge_predict'))
            #

            imshow(title='Train', imgs=(imgs[0, dataset.img_channels // 2], labels[0], predicts_vis[0], teeth_edges[0], predicts_boundary_teeth_vis[0],root_edges[0],predicts_boundary_root_vis[0]),
                   shape=(1, 7), subtitle=('image', 'label', 'predict','boundaryteeth','pre_teeth_edge','boundaryroot',  'pre_root_edge'))
        losses['total'] = loss
        for k in losses.keys(): losses[k] = losses[k].item()
        tbar.set_postfix(losses)
    
    scheduler.step(loss.item())
    
    for k, v in losses.items():
        logger.add_scalar(f'loss/{k}', v, epoch)
    
    return loss.item()


def evaluation(net, dataset, epoch, batch_size, num_workers, vis_intvl, logger, type):
    type = type.lower()
    if type == 'train':
        subset = dataset.train_dataset
        case_slice_indices = dataset.train_case_slice_indices
    elif type == 'valid':
        subset = dataset.valid_dataset
        case_slice_indices = dataset.valid_case_slice_indices
    
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)
    evaluator = Evaluator(dataset.num_classes)
    
    case = 0
    vol_label = []
    vol_output = []
    
    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, labels, idx = data['image'].cuda(), data['label'], data['index']
            
            outputs = net(imgs)
            predicts = outputs['output']
            predicts = predicts.argmax(dim=1)
            
            labels = labels.cpu().detach().numpy()
            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()
            
            vol_label.append(labels)
            vol_output.append(predicts)
            
            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_label = np.concatenate(vol_label, axis=0)
                
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]
                evaluator.add(vol_output[:vol_num_slice], vol_label[:vol_num_slice])
                
                vol_output = [vol_output[vol_num_slice:]]
                vol_label = [vol_label[vol_num_slice:]]
                case += 1
                pbar.update(1)
            
            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                data['predict'] = predicts
                data = dataset.vis_transform(data)
                imgs, labels, predicts = data['image'], data['label'], data['predict']
                imshow(title=f'eval/{type:5}', imgs=(imgs[0, dataset.img_channels // 2], labels[0], predicts[0]),
                       shape=(1, 3), subtitle=('image', 'label', 'predict'))
    
    acc = evaluator.eval()
    
    for k in sorted(list(acc.keys())):
        if k == 'dc_each_case': continue
        print(f'{type}/{k}: {acc[k]:.5f}')
        logger.add_scalar(f'{type}_acc_total/{k}', acc[k], epoch)
    
    for case_idx in range(len(acc['dc_each_case'])):
        case_id = dataset.case_idx_to_case_id(case_idx, type)
        dc_each_case = acc['dc_each_case'][case_idx]
        for cls in range(len(dc_each_case)):
            dc = dc_each_case[cls]
            logger.add_scalar(f'{type}_acc_each_case/case_{case_id:05d}/dc_{cls}', dc, epoch)
    
    score = (acc['dc_per_case_1'] + acc['dc_per_case_2']) / 2
    logger.add_scalar(f'{type}/score', score, epoch)
    return score

def dice_edge_loss(y_true, y_pred):
    smooth = 0.001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def find_boundaries(label_img, connectivity=1, mode='thick', background=0):
    """Return bool array where boundaries between labeled regions are True.
    """

    if label_img.dtype == 'bool': #predicts_teeth_area[i] [256,256]
        label_img = label_img.astype(np.uint8)
    ndim = label_img.ndim #2
    selem = ndi.generate_binary_structure(ndim, connectivity)
    if mode != 'subpixel':
        boundaries = dilation(label_img, selem) != erosion(label_img, selem)
        if mode == 'inner':
            foreground_image = (label_img != background)
            boundaries &= foreground_image
        elif mode == 'outer':
            max_label = np.iinfo(label_img.dtype).max
            background_image = (label_img == background)
            selem = ndi.generate_binary_structure(ndim, ndim)
            inverted_background = np.array(label_img, copy=True)
            inverted_background[background_image] = max_label
            adjacent_objects = ((dilation(label_img, selem) !=
                                 erosion(inverted_background, selem)) &
                                ~background_image)
            boundaries &= (background_image | adjacent_objects)
        return boundaries
    else:
        boundaries = _find_boundaries_subpixel(label_img)
        return boundaries

def _find_boundaries_subpixel(label_img):

    ndim = label_img.ndim
    max_label = np.iinfo(label_img.dtype).max

    label_img_expanded = np.zeros([(2 * s - 1) for s in label_img.shape],
                                  label_img.dtype)
    pixels = (slice(None, None, 2), ) * ndim
    label_img_expanded[pixels] = label_img

    edges = np.ones(label_img_expanded.shape, dtype=bool)
    edges[pixels] = False
    label_img_expanded[edges] = max_label
    windows = view_as_windows(np.pad(label_img_expanded, 1,
                                     mode='constant', constant_values=0),
                              (3,) * ndim)

    boundaries = np.zeros_like(edges)
    for index in np.ndindex(label_img_expanded.shape):
        if edges[index]:
            values = np.unique(windows[index].ravel())
            if len(values) > 2:  # single value and max_label
                boundaries[index] = True
    return boundaries


if __name__ == '__main__':
    main()
