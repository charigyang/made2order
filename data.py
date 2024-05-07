from datasets.dataset import *
from torch.utils.data import Dataset, DataLoader
import utils
def get_dataset(args):
    if args.dataset == 'mnist':
        data = MultiDigitSplits(dataset=args.dataset, num_compare=args.num_compare)
        data_loader_train = data.get_train_loader(batch_size=args.batch_size, drop_last=True)
        data_loader_valid = data.get_valid_loader(batch_size=args.batch_size, drop_last=True)
        image_size = (28, 112)
        channels = 1
        image_patch_size = 7

    if args.dataset == 'svhn':
        data = MultiDigitSplits(dataset=args.dataset, num_compare=args.num_compare)
        data_loader_train = data.get_train_loader(batch_size=args.batch_size, drop_last=True)
        data_loader_valid = data.get_test_loader(batch_size=args.batch_size, drop_last=True)
        image_size = (54, 54)
        channels = 3
        image_patch_size = 6

    if args.dataset == 'rds':
        train_dataset = RandomDotsDataset(args.num_compare, args.reversible, 100000)
        val_dataset = RandomDotsDataset(args.num_compare, args.reversible, 1000)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=2, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=2, pin_memory=True, shuffle=True)
        image_size = 42
        channels = 1    
        image_patch_size = 7

    if args.dataset == 'clocks_cropped':
        train_dataset = TimelapseClocks(args.num_compare, 100000,True)
        val_dataset = TimelapseClocks(args.num_compare, 1000,False)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True)
        image_size = 196
        channels = 3
        image_patch_size = 14

    if args.dataset == 'clocks_full':
        train_dataset = TimelaspseFull(train=True, t=args.num_compare, dt=5)
        val_dataset = TimelaspseFull(train=False, t=args.num_compare, dt=5)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True)
        image_size = (160*2, 240*2)
        channels = 3 
        image_patch_size = 20

    if args.dataset == 'scenes':
        train_dataset = SkyLapseDataset(num_compare=args.num_compare, train=True)
        val_dataset = SkyLapseDataset(num_compare=args.num_compare, train=False)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True)
        image_size = (336, 336)
        channels = 3 
        image_patch_size = 21

    if args.dataset == 'moca':
        train_dataset = MoCADataset(num_compare=args.num_compare, train=True)
        val_dataset = MoCADataset(num_compare=args.num_compare, train=False)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=1, pin_memory=False, shuffle=True)
        image_size = (336, 336)
        channels = 3 
        image_patch_size = 21

    if args.dataset == 'muds':
        train_dataset = SpaceDataset(train=True, t=args.num_compare)
        val_dataset = SpaceMonoDataset(train=False, t=args.num_compare)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=60, drop_last=False, num_workers=8, pin_memory=True, shuffle=False)
        image_size = (196,196)
        channels = 3 
        image_patch_size = 7

    if args.dataset == 'calfire':
        train_dataset = CalFireDataset(train=True, t=args.num_compare)
        val_dataset = CalFireDataset(train=False, t=args.num_compare)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True)
        image_size = (196,196)
        channels = 3 
        image_patch_size = 14

    if args.dataset == 'mri3':
        train_dataset = MRI3Dataset(train=True, t=args.num_compare)
        val_dataset = MRI3Dataset(train=False, t=args.num_compare)
        data_loader_train = utils.FastDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
        data_loader_valid = utils.FastDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True)
        image_size = (224, 154)
        channels = 1
        image_patch_size = 14

    image_size = utils.pair(image_size)
    return data_loader_train, data_loader_valid, image_size, channels, image_patch_size