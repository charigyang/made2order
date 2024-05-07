import collections
import functools
import operator
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
import os
import cv2
import einops



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def ranking_accuracy(model, data, targets, reversible=False):
    scores, attns = model(data)
    _, scores = torch.max(scores, 2)
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    if reversible:
        acc_r =  torch.argsort(targets, dim=-1, descending=True) == torch.argsort(scores, dim=-1)
        acc = acc + acc_r
    acc_em = acc.all(-1).float().mean()
    acc_ew = acc.float().mean()

    # EM5:
    scores, attns = model(data[:,:5])
    _, scores = torch.max(scores, 2)
    targets = targets[:, :5]
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    if reversible:
        acc_r =  torch.argsort(targets, dim=-1, descending=True) == torch.argsort(scores, dim=-1)
        acc = acc + acc_r
    acc_em5 = acc.all(-1).float().mean()
    acc_ew5 = acc.float().mean()

    return dict(
        acc_em=acc_em.type(torch.float64).mean().item(),
        acc_ew=acc_ew.type(torch.float64).mean().item(),
        acc_em5=acc_em5.type(torch.float64).mean().item(),
        acc_ew5=acc_ew5.type(torch.float64).mean().item(),
    )

def compute_iou(y_true, y_pred, threshold = 0.05):

    y_pred_binary = (y_pred > threshold).float()
    # Assuming y_true and y_pred are batched tensors of shape (batch_size, height, width)
    intersection = torch.logical_and(y_true, y_pred)
    union = torch.logical_or(y_true, y_pred)
    
    # Sum along the (height, width) dimensions
    intersection = torch.sum(intersection, dim=(-2, -1))
    union = torch.sum(union, dim=(-2, -1))
    
    iou = intersection / union
    return iou

def ranking_accuracy_seg(model, data, targets, gt, reversible=False):
    scores, attns = model(data)
    _, scores = torch.max(scores, 2)
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    if reversible:
        acc_r =  torch.argsort(targets, dim=-1, descending=True) == torch.argsort(scores, dim=-1)
        acc = acc + acc_r
    acc_em = acc.all(-1).float().mean()
    acc_ew = acc.float().mean()

    bsz = 32
    psz = 21
    imsz = 336
    w = imsz // psz
    thres = 0.01

    # EM5:
    #gt: b,h,w
    attns = einops.reduce(attns, 'b f n q -> b n', 'mean')
    argmax_indices = torch.argmax(attns, dim=1)
    argy = argmax_indices // w * psz + (psz//2)
    argx = argmax_indices % w * psz + (psz//2)


    accs = []
    for i in range(bsz):
        accs.append(gt[i,argy[i],argx[i]])


    attns = einops.rearrange(attns, 'b (h w) -> b h w', h = w)
    attn = F.interpolate(attns.unsqueeze(1), size=(imsz, imsz), mode='nearest').squeeze(1)
    iou = compute_iou(gt, attn, threshold = thres)

    acc_em5 = sum(accs)/len(accs)
    acc_ew5 = iou.mean()

    return dict(
        acc_em=acc_em.type(torch.float64).mean().item(),
        acc_ew=acc_ew.type(torch.float64).mean().item(),
        acc_em5=acc_em5.type(torch.float64).mean().item(),
        acc_ew5=acc_ew5.type(torch.float64).mean().item(),
    )

def ranking_accuracy_mono(model, data, targets, reversible=False):
    scores, attns = model(data) #scores b f q , attns b f n q

    _, scores = torch.max(scores, 2)
    target = torch.tensor([0, 1, 2, 3]).cuda()
    contains_all_targets = torch.all(torch.eq(torch.sort(scores, dim=1)[0][:, None], target), dim=2)
    count_rows_with_all_targets = torch.sum(contains_all_targets, dim=1)
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    if reversible:
        acc_r =  torch.argsort(targets, dim=-1, descending=True) == torch.argsort(scores, dim=-1)
        acc = acc + acc_r
    acc_em = acc.all(-1).float().mean()
    acc_ew = acc.float().mean()



    #pg
    gt_dir = 'Monotonic_MUDS/masks'
    gts = natsorted([x for x in os.listdir(gt_dir) if 'png' in x])
    gt_list = []
    for gt in gts:
        gt_list.append(cv2.imread(os.path.join(gt_dir, gt))[:,:,0:1])
    
    order = torch.argsort(scores, dim=-1) #b f
    
    num_frames = 4
    attns = einops.rearrange(attns, 'b f (h w) q -> b f q h w', h=196//7).detach().cpu().numpy()
    to_sam = []
    for i in range(len(gt_list)):
        gt = gt_list[i]
        amap0 = attns[i, order[i,0],0]
        amap1 = attns[i, order[i,1],1]
        amap2 = attns[i, order[i,2],2]
        amap3 = attns[i, order[i,3],3] #h28,w28 each
        to_sam.append(np.stack([amap0,amap1,amap2,amap3], 0)) #4,h,w
        #just pointing game and save masks from here!
    sam = np.stack(to_sam, 0) #60 4 h w
    sam_disp = einops.rearrange(sam, 'n f h w -> (n h) (f w)')
        

    # EM5:

    acc = (targets == scores)
    acc_em5 = acc.all(-1).float().mean()
    acc_ew5 = count_rows_with_all_targets.float().mean()


    return dict(
        acc_em=acc_em.type(torch.float64).mean().item(),
        acc_ew=acc_ew.type(torch.float64).mean().item(),
        acc_em5=acc_em5.type(torch.float64).mean().item(),
        acc_ew5=acc_ew5.type(torch.float64).mean().item(),
    ), sam_disp


def avg_list_of_dicts(list_of_dicts):
    result = {}
    for k in list_of_dicts[0]:
        result[k] = np.mean([d[k] for d in list_of_dicts])
    return result


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)