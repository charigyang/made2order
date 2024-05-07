import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import InterpolationMode
import torchvision.io as tvio
import torchvision.transforms.functional as fn

import imageio as imageio
import numpy as np
import pandas as pd
import random
import einops
from einops import rearrange
import io
import os
import os.path
import cv2
from PIL import Image
import nibabel as nib
from natsort import natsorted

def dot_on_canvas(x,y,sz=42, dot_sz=3,rds=False, rds_box=None,canvas_box=None):
    if rds:
        canvas = canvas_box.clone()
        canvas[y-4:y+5, x-4:x+5] = rds_box
    else:
        canvas = torch.zeros([42, 42])
        canvas[y-2:y+3, x-2:x+3] = 255.
    return canvas

def dot_on_canvas_set(sz, x=16, y=16, dx=1, dy=1, num_compare=5, rds=False):
    Is=[]
    canvas_box = torch.randint(low=0, high=2, size=(sz, sz)) * 255.
    rds_box = torch.randint(low=0, high=2, size=(9, 9)) * 255.
    for i in range(num_compare):
        Is.append(dot_on_canvas(x%sz,y%sz, rds=rds, rds_box=rds_box, canvas_box=canvas_box))
        x=x+dx
        y=y+dy
    return Is 

def dot_on_canvas_gt(sz, bsz=9, x=16, y=16, dx=1, dy=1, num_compare=5, rds=False):
    Is=[]
    bg = torch.zeros(sz,sz)
    for i in range(num_compare):
        bg[y+i*dy:y+i*dy+bsz,x+i*dx:x+i*dx+bsz] = 1.
    return bg


class RandomDotsDataset(Dataset):
    def __init__(self, num_compare, reverse=False, len=100000, local=True):
        self.num_compare = num_compare
        self.len = len
        self.reverse = reverse
        self.local=local

    def __len__(self):
            return self.len
    
    def __getitem__(self, idk):
        if self.reverse:
            dx = np.random.choice([1,2,3,-1,-2,-3])#,-1,-2,-3])
            dy = np.random.choice([1,2,3,-1,-2,-3])#,-1,-2,-3])
        else: 
            dx = np.random.choice([1,2,3])#,-1,-2,-3])
            dy = np.random.choice([1,2,3])#,-1,-2,-3])
        num = self.num_compare -1
        sz = 42
        if dx>0:
            x = np.random.randint(4, sz-4 - (num*dx))
        else:
            x = np.random.randint(4 - (num*dx), sz-4)
        if dy>0:
            y = np.random.randint(4, sz-4 - (num*dy))
        else:
            y = np.random.randint(4 - (num*dy), sz-4)

        Is = dot_on_canvas_set(sz, x,y,dx,dy, num_compare=self.num_compare, rds=True)
        gt = dot_on_canvas_gt(sz, 9, x,y,dx,dy, num_compare=self.num_compare, rds=False)
        targets = list(range(self.num_compare))
        random.shuffle(targets)
        
        if not self.local:
            return torch.stack([Is[x] for x in targets], 0).unsqueeze(1), torch.Tensor(targets)
        else:
            return torch.stack([Is[x] for x in targets], 0).unsqueeze(1), torch.Tensor(targets), gt

class TimelapseClocks(Dataset):
    def __init__(self, num_compare, len=100000, train=True):
        self.num_compare = num_compare
        self.len = len
        self.data_dir = '/scratch/shared/beegfs/charig/time/WebVid/cropped_data_cbnet/'
        self.data_list = '/work/charig/current/pose/tlapse_out.txt'
        with open(self.data_list, 'r') as file:
            self.videos = [line.rstrip('\n') for line in file.readlines()]
        self.videos = self.videos[:-500] if train else self.videos[-500:]
        self.train=train
        if self.train:
            self.transform = transforms.Compose([transforms.RandomCrop((196, 196)),])
        else:
            self.transform = transforms.Compose([transforms.CenterCrop((196, 196)),])

    def __len__(self):
            return len(self.videos)
    
    def __getitem__(self, idx):
        ims = sorted(os.listdir(self.data_dir + self.videos[idx]))
        num_frames = len(ims)
        
        dt = np.clip(np.random.randint(10,30),0,num_frames//5)
        high = max(1,num_frames - dt*(self.num_compare-1))
        start_frame = np.random.randint(low=0, high=high)

        end_frame = start_frame + dt*(self.num_compare-1)
        frame_indices = np.arange(start_frame, end_frame+1, dt)

        np.random.shuffle(frame_indices)
        out = torch.stack([torch.Tensor(cv2.resize(cv2.imread(os.path.join(self.data_dir, self.videos[idx], ims[i])), (224, 224))) for i in frame_indices], 0)
        out = einops.rearrange(out, 'f h w c -> f c h w')
        out = self.transform(out)
        return out, frame_indices



class TimelaspseFull(Dataset):
    def __init__(self, train=True, t=5, dt=10):
        self.t = t
        self.dt = dt
        
        self.data_dir = '/scratch/shared/beegfs/charig/order/static_lapse_jpg'
        self.videos = sorted(os.listdir(self.data_dir))

        #shuffle(self.videos)
        self.frames = self.videos[:-50] if train else self.videos[-50:]
        self.train=train
        print(len(self.videos))

        if train:
            t = random.choice([InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT, InterpolationMode.BICUBIC])
            self.transform = transforms.Compose([transforms.RandomResizedCrop(size=(160*2, 240*2), scale=(0.6,0.9), ratio=(0.9,1.1),interpolation=t),])
        else:
            self.transform = transforms.Compose([transforms.CenterCrop((160*2, 240*2)),])
        

    def __len__(self):
        return len(self.frames) * 10
    

    def __getitem__(self, idx):
        idx = idx % len(self.frames)
        frames = sorted(os.listdir(os.path.join(self.data_dir,self.frames[idx])))
        num_frames = len(frames)
        
        #dt = np.random.randint(8, 13) orig . num_frames > dt * (t-1)
        dt = np.clip(np.random.randint(30, 150), 0, num_frames // (self.t)) #30 150
        high = max(1,num_frames - dt*(self.t-1))
        start_frame = np.random.randint(low=0, high=high)
        end_frame = start_frame + dt*(self.t-1)
        frame_indices = np.arange(start_frame, end_frame+1, dt)
        dt2 = np.random.randint(-10, 10, self.t)
        frame_indices = np.clip(frame_indices + dt2, 0, num_frames-1)
        
        np.random.shuffle(frame_indices)
        
        ims = []
        for i in frame_indices:
            ims.append(cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, self.frames[idx], frames[i])), cv2.COLOR_BGR2RGB))
        

        # apply transform
        ims = torch.Tensor(np.stack(ims, 0))
        ims = rearrange(ims,'f h w c -> f c h w')      
        ims = self.transform(ims)

        return ims/1., frame_indices


class SkyLapseDataset(Dataset):
    def __init__(self, num_compare, length=100000, train=True):
        self.num_compare = num_compare
        self.len = length
        self.data_dir = '/scratch/shared/beegfs/charig/order/timelapsesky/jpg'
        self.videos = sorted(os.listdir(self.data_dir))
        self.filter_dir = '/scratch/shared/beegfs/charig/order/timelapsesky/setvak'
        self.videos = sorted([x.split('.')[0] for x in os.listdir(self.filter_dir) if ".mov" not in x])
        #shuffle(self.videos)
        self.frames = self.videos[:130] if train else self.videos[130:]
        self.train=train
        if self.train:
            self.transform = transforms.Compose([transforms.RandomCrop((336, 336)),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.GaussianBlur(7, (0.1,2.0)),
                                            ])
        else:
            self.transform = transforms.Compose([transforms.CenterCrop((336, 336)),])

    def __len__(self):
            return len(self.frames) * 10
    
    def __getitem__(self, idx):
        idx = idx % len(self.frames)
        frames = sorted(os.listdir(os.path.join(self.data_dir,self.frames[idx])))
        num_frames = len(frames)

        dt = np.clip(np.random.randint(30, 150), 0, num_frames // (self.num_compare))
        high = max(1,num_frames - dt*(self.num_compare-1))
        start_frame = np.random.randint(low=0, high=high)

        end_frame = start_frame + dt*(self.num_compare-1)
        frame_indices = np.arange(start_frame, end_frame+1, dt)

        np.random.shuffle(frame_indices)
        ims = []
        for i in frame_indices:
            ims.append(cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, self.frames[idx], frames[i])), cv2.COLOR_BGR2RGB))

        # apply transform
        if self.transform:
            ims = torch.Tensor(np.stack(ims, 0))
            ims = rearrange(ims,'f h w c -> f c h w')
            if self.train: ims = ims + torch.randn_like(ims) * 3
            ims = self.transform(ims)
            if self.train: ims = ims + torch.randn_like(ims) * 3
            ims = torch.clip(ims, 0, 255.)
        return ims/1., frame_indices

class SpaceDataset(Dataset):
    def __init__(self, train=True, t=5, dt=10):
        self.t = t
        self.dt = dt
        
        if train:
            self.data_dir = '/scratch/shared/beegfs/charig/order/SN7_buildings_train/train'
        else:
            self.data_dir = '/scratch/shared/beegfs/charig/order/SN7_buildings_test_public/test_public'
        self.videos = natsorted(os.listdir(self.data_dir))

        self.frames = self.videos

        self.train=train 
        self.im_choice = 'images' if train else 'images_masked'

        if self.train:
            self.transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            ])

    def __len__(self):
        return len(self.frames) * 10
    

    def __getitem__(self, idx):
        idx = idx % len(self.frames)
        frames = sorted(os.listdir(os.path.join(self.data_dir,self.frames[idx], 'images_masked')))
        num_frames = len(frames)

        dt = np.random.randint(2, 6)
        high = max(1,num_frames - dt*(self.t-1))
        start_frame = np.random.randint(low=0, high=high)
        end_frame = start_frame + dt*(self.t-1)
        frame_indices = np.arange(start_frame, end_frame+1, dt)

        np.random.shuffle(frame_indices)
        x = np.random.randint(0, 1024-196)
        y = np.random.randint(0, 1024-196)

        ims = []
        for i in frame_indices:
            ims.append(np.asarray(Image.open(os.path.join(self.data_dir, self.frames[idx], self.im_choice, frames[i])).crop((x,y,x+196,y+196)))[:,:,:3])

        # apply transform
        ims = torch.Tensor(np.stack(ims, 0))
        ims = rearrange(ims,'f h w c -> f c h w')      
        if self.train:
            ims = self.transform(ims)
        return ims/1., frame_indices


class SpaceMonoDataset(Dataset):
    def __init__(self, train=False, t=4, dt=0):        
        self.data_dir = 'Monotonic_MUDS/img_all'
        self.frames = natsorted([x for x in os.listdir(self.data_dir) if 'png' in x])

    def __len__(self):
        return len(self.frames) // 4
    

    def __getitem__(self, idx):
        frames = self.frames
        frames = frames[idx*4:(idx+1)*4]
        frame_indices = np.array([0,1,2,3])
        np.random.shuffle(frame_indices)
        ims = []

        for i in frame_indices:
            ims.append(cv2.imread(os.path.join(self.data_dir, frames[i]))[:,:,::-1])

        # apply transform
        ims = torch.Tensor(np.stack(ims, 0))
        ims = rearrange(ims,'f h w c -> f c h w')      
        return ims/1., frame_indices


class CalFireDataset(Dataset):
    def __init__(self, train=True, t=4, dt=10):
        self.t = t
        self.dt = dt
        
        self.data_dir = '/scratch/shared/beegfs/charig/calfire2/CalFire/fulldata'
        self.videos = sorted(os.listdir(self.data_dir))
        vids = []
        for v in self.videos:
            frames  = sorted(os.listdir(os.path.join(self.data_dir,v)))
            n = [x for x in frames if 'cloud' not in x]
            len_valid = len(n)
            if len_valid>=4:
                vids.append(v)
        print(len(self.videos))
        if train:
            self.videos = vids[:-200]
        else: 
            self.videos = vids[-200:]

        self.frames = self.videos
        self.train=train 

        if self.train:
            self.transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomResizedCrop(size=(196,196))
                                            ])


    def __len__(self):
        return len(self.frames) #* 10
    

    def __getitem__(self, idx):
        frames = sorted(os.listdir(os.path.join(self.data_dir,self.frames[idx])))
        frames = [x for x in frames if 'cloud' not in x]
        num_frames = len(frames)

        frame_indices = np.random.choice(range(num_frames), self.t, replace=False)

        np.random.shuffle(frame_indices)
        x = np.random.randint(0, 1024-336)
        y = np.random.randint(0, 1024-336)

        ims = []
        for i in frame_indices:
            img = cv2.imread(os.path.join(self.data_dir, self.frames[idx], frames[i]))
            ims.append(cv2.resize(img[:,:,::-1], (196,196)))

        # apply transform
        ims = torch.Tensor(np.stack(ims, 0))
        ims = rearrange(ims,'f h w c -> f c h w')      
        if self.train:
            ims = self.transform(ims)
        return ims/1., frame_indices


class MRI3Dataset(Dataset):
    def __init__(self, train=True, t=4, dt=10):
        self.t = t
        self.dt = dt
        
        self.data_dir = '/scratch/shared/beegfs/charig/oasis/images'
        self.videos = list(set([x.split('_')[0] for x in sorted(os.listdir(self.data_dir))]))
        
        if train:
            self.videos = self.videos[:100]
        else: 
            self.videos = self.videos[100:] #134

        self.frames = self.videos
        self.train=train 

        if self.train:
            self.transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            ])

    def __len__(self):
        return len(self.frames) * 10
    

    def __getitem__(self, idx):
        idx = idx % len(self.frames)
        frames = [x for x in sorted(os.listdir(self.data_dir)) if self.frames[idx] in x]#
        num_frames = len(frames)

        frame_indices = np.random.choice(range(num_frames), self.t, replace=False)

        if self.train:
            x = np.random.randint(0, 32)
            y = np.random.randint(0, 32)
            y2 = np.random.randint(256-32, 256)
            x2 = np.random.randint(176-32, 176)
        else:
            x = 0
            y = 0
            x2 = 176
            y2 = 256
        

        ims = []

        for i in list(frame_indices):
            img = cv2.imread(os.path.join(self.data_dir, frames[i]))[y:y2,x:x2,0]
            img = cv2.resize(img, (154,224))

            ims.append(img)

        # apply transform
        ims = torch.Tensor(np.stack(ims, 0))
        ims = rearrange(ims,'f h w -> f 1 h w')

        return ims , frame_indices

class MoCADataset(Dataset):
    def __init__(self, num_compare, length=100000, train=True):
        self.num_compare = num_compare
        self.len = length
        self.data_dir = '/scratch/shared/beegfs/charig/MoCA_filtered/JPEGImages_360p'
        self.videos = sorted(os.listdir(self.data_dir))

        self.frames = self.videos[:75] if train else self.videos[75:]
        self.train=train
        self.cva = []

        if self.train:
            
            t = random.choice([InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT, InterpolationMode.BICUBIC])
            self.transform = transforms.Compose([transforms.RandomResizedCrop(size=(336, 336), scale=(0.6,0.9), ratio=(0.9,1.1),interpolation=t),
                                                transforms.RandomHorizontalFlip(p=0.5),])
        else:
            self.transform = transforms.Compose([transforms.CenterCrop((336, 336)),])

    def __len__(self):
            return len(self.frames) * 10
    

    def __getitem__(self, idx):
        idx = idx % len(self.frames)
        frames = sorted(os.listdir(os.path.join(self.data_dir,self.frames[idx])))
        num_frames = len(frames)
        
        dt = random.choice([1,2])
        high = max(1,num_frames - dt*(self.num_compare-1))
        start_frame = np.random.randint(low=0, high=high)

        end_frame = start_frame + dt*(self.num_compare-1)
        frame_indices = np.arange(start_frame, end_frame+1, dt)

        np.random.shuffle(frame_indices)
        ims = []
        for i in frame_indices:
            ims.append(cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, self.frames[idx], frames[i])), cv2.COLOR_BGR2RGB))

        # apply transform
        if self.transform:
            ims = torch.Tensor(np.stack(ims, 0))
            ims = rearrange(ims,'f h w c -> f c h w')
            if self.train:
                x = np.random.randint(5, 640-336-5+1)
                y = np.random.randint(5, 360-336-5+1)
                dx = np.random.randint(-5,5+1,5)
                dy = np.random.randint(-5,5+1,5)
                ims_ = []
                for i in range(self.num_compare):
                    ims_.append(ims[i,:,y+dy[i]:y+dy[i]+336, x+dx[i]:x+dx[i]+336])
                ims = torch.stack(ims_,0)
            
            ims = self.transform(ims)
        
        if not self.train:
            gts=[]
            gt_dir = '/scratch/shared/beegfs/charig/MoCA_filtered/Annotations'
            for i in frame_indices:
                gts.append(cv2.imread(os.path.join(gt_dir, self.frames[idx], frames[i].replace('jpg', 'png'))))
            gts = torch.Tensor(np.stack(gts, 0))
            gts = rearrange(gts,'f h w c -> f c h w') /255.
            gts = self.transform(gts)
            gts = einops.reduce(gts, 'f c h w -> h w', 'mean')
            gts = (gts > 0.1).float()
            
            diff_img = torch.sqrt(torch.sum((ims[0] - ims[-1]) ** 2, 0)) #chw
            idx = torch.argmax(diff_img)
            y = idx // 336
            x = idx % 336
            self.cva.append(gts[y,x].item())
        else:
            gts = ims

        return ims/1., frame_indices, gts


# Copyright (c) Felix Petersen.
class MultiDigitSplits(object):
    def __init__(self, dataset, num_digits=4, num_compare=None, seed=0, deterministic_data_loader=True):

        self.deterministic_data_loader = deterministic_data_loader

        if dataset == 'mnist':
            trva_real = datasets.MNIST(root='./data-mnist', download=True)
            xtr_real = trva_real.data[:55000].view(-1, 1, 28, 28)
            ytr_real = trva_real.targets[:55000]
            xva_real = trva_real.data[55000:].view(-1, 1, 28, 28)
            yva_real = trva_real.targets[55000:]

            te_real = datasets.MNIST(root='./data-mnist', train=False, download=True)
            xte_real = te_real.data.view(-1, 1, 28, 28)
            yte_real = te_real.targets

            self.train_dataset = MultiDigitDataset(
                images=xtr_real, labels=ytr_real, num_digits=num_digits, num_compare=num_compare, seed=seed,
                determinism=deterministic_data_loader)
            self.valid_dataset = MultiDigitDataset(
                images=xva_real, labels=yva_real, num_digits=num_digits, num_compare=num_compare, seed=seed)
            self.test_dataset = MultiDigitDataset(
                images=xte_real, labels=yte_real, num_digits=num_digits, num_compare=num_compare, seed=seed)

        elif dataset == 'svhn':
            self.train_dataset = SVHNMultiDigit(root='./data-svhn', split='train', num_compare = num_compare, download=True)
            self.valid_dataset = SVHNMultiDigit(root='./data-svhn', split='val', num_compare = num_compare, download=True)
            self.test_dataset = SVHNMultiDigit(root='./data-svhn', split='test', num_compare = num_compare, download=True)
        else:
            raise NotImplementedError()

    def get_train_loader(self, batch_size, **kwargs):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  num_workers=4 if not self.deterministic_data_loader else 0,
                                  shuffle=True, **kwargs)
        return train_loader

    def get_valid_loader(self, batch_size, **kwargs):
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size, shuffle=False, **kwargs)
        return valid_loader

    def get_test_loader(self, batch_size, **kwargs):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size, shuffle=False, **kwargs)
        return test_loader

# Copyright (c) Felix Petersen.
class SVHNMultiDigit(VisionDataset):
    """`Preprocessed SVHN-Multi <>`_ Dataset.
    Note: The preprocessed SVHN dataset is based on the the `Format 1` official dataset.
    By cropping the numbers from the images, adding a margin of :math:`30\%` , and resizing to :math:`64\times64` ,
    the dataset has been preprocessed.
    The data split is as follows:

        * ``train``: (30402 of 33402 original ``train``) + (200353 of 202353 original ``extra``)
        * ``val``: (3000 of 33402 original ``train``) + (2000 of 202353 original ``extra``)
        * ``test``: (all of 13068 original ``test``)

    Each ```train / val`` split has been performed using
    ``sklearn.model_selection import train_test_split(data_X_y_tuples, test_size=3000 / 2000, random_state=0)`` .
    This is the closest that we could come to the
    `work by Goodfellow et al. 2013 <https://arxiv.org/pdf/1312.6082.pdf>`_ .

    Args:
        root (string): Root directory of dataset where directory
            ``SVHNMultiDigit`` exists.
        split (string): One of {'train', 'val', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`` .
            (default = random 54x54 crop + normalization)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    split_list = {
        'train': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_train.p",
                  "svhn-multi-digit-3x64x64_train.p", "25df8732e1f16fef945c3d9a47c99c1a"],
        'val': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_val.p",
                "svhn-multi-digit-3x64x64_val.p", "fe5a3b450ce09481b68d7505d00715b3"],
        'test': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_test.p",
                 "svhn-multi-digit-3x64x64_test.p", "332977317a21e9f1f5afe7ef47729c5c"]
    }

    def __init__(self, root, split='train',num_compare=5,
                 transform=transforms.Compose([
                     transforms.RandomCrop([54, 54]),
                     transforms.ToTensor(),
                     #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ]),
                 target_transform=None, download=False):
        super(SVHNMultiDigit, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data = torch.load(os.path.join(self.root, self.filename))

        self.data = data[0]
        # loading gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = data[1].type(torch.LongTensor)
        self.num_compare = num_compare

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        indices = np.random.randint(0, len(self.data), self.num_compare)
        imgs = []
        targets = []
        for index in indices:
            img, target = self.data[index], int(self.labels[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs, 0) * 255., torch.Tensor(targets)

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


class MultiDigitDataset(Dataset):
    def __init__(
            self,
            images,
            labels,
            num_digits,
            num_compare,
            seed=0,
            determinism=True,
    ):
        super(MultiDigitDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.num_digits = num_digits
        self.num_compare = num_compare
        self.seed = seed
        self.rand_state = None

        self.determinism = determinism

        if determinism:
            self.reset_rand_state()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        if self.determinism:
            prev_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.rand_state)

        labels = []
        images = []
        labels_ = None
        for digit_idx in range(self.num_digits):
            id = torch.randint(len(self), (self.num_compare, ))
            labels.append(self.labels[id])
            images.append(self.images[id].type(torch.float32))
            if labels_ is None:
                labels_ = torch.zeros_like(labels[0] * 1.)
            labels_ = labels_ + 10.**(self.num_digits - 1 - digit_idx) * self.labels[id]

        images = torch.cat(images, dim=-1)


        if self.determinism:
            self.rand_state = torch.random.get_rng_state()
            torch.random.set_rng_state(prev_state)

        return images, labels_

    def reset_rand_state(self):
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        self.rand_state = torch.random.get_rng_state()
        torch.random.set_rng_state(prev_state)