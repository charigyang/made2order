import einops
import argparse
from tqdm import tqdm
import random
import torch
import utils
import numpy as np
from data import get_dataset
from tensorboardX import SummaryWriter
from datetime import datetime
from model import Made2Order


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-n', '--num_compare', type=int, default=5)
    parser.add_argument('-i', '--num_steps', type=int, default=100000, help='number of training steps')
    parser.add_argument('-e', '--eval_freq', type=int, default=1000, help='the evaluation frequency')
    parser.add_argument('-d', '--dataset', type=str, default='mnist')#, choices=['mnist', 'clock', 'pose', 'sky', 'dots', 'lapse', 'rdots', 'rds'])
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-v', '--verbose', type=str, default='', help='verbose')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--reversible', action='store_true', default=False)
    args = parser.parse_args()

    #logging
    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    writer = SummaryWriter(logdir='logs/{}-{}'.format(dt_string, args.verbose))
    
    #data and model
    data_loader_train, data_loader_valid, image_size, channels, image_patch_size = get_dataset(args)
    model = Made2Order(image_size=image_size, image_patch_size=image_patch_size,channels=channels, frames=args.num_compare).to(args.device)

    lr = args.lr
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    valid_accs = []
    train_ems = []
    train_ews = []
    losses = []

    for iter_idx, (data, targets) in tqdm(
        enumerate(utils.load_n(data_loader_train, args.num_steps)),
        desc="Training steps",
        total=args.num_steps,
    ):
        data = data.to(args.device)
        targets = targets.to(args.device)
        perm_prediction, attns = model(data)
        
        #loss
        if args.reversible:
            perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1)).transpose(-2, -1).float()
            loss1 = torch.nn.BCELoss(reduction='none')(torch.nn.functional.softmax(perm_prediction, -1), perm_ground_truth).mean(1).mean(1)
            perm_ground_truth2 = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1, descending=True)).transpose(-2, -1).float()
            loss2 = torch.nn.BCELoss(reduction='none')(torch.nn.functional.softmax(perm_prediction, -1), perm_ground_truth2).mean(1).mean(1)
            loss = torch.mean(torch.minimum(loss1,loss2))
        else:
            perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1)).transpose(-2, -1).float()
            loss = torch.nn.BCELoss()(torch.nn.functional.softmax(perm_prediction, -1), perm_ground_truth)    

        if iter_idx < 100:
            lr_ = lr * iter_idx / 100
            for g in optim.param_groups:
                g['lr'] = lr_

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        #training accuracy
        _, scores = torch.max(perm_prediction, 2)
        acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
        if args.reversible:
            acc_r =  torch.argsort(targets, dim=-1, descending=True) == torch.argsort(scores, dim=-1)
            acc = acc + acc_r
        acc_em = acc.all(-1).float().mean()
        acc_ew = acc.float().mean()

        train_ems.append(acc_em)
        train_ews.append(acc_ew)
        losses.append(loss)

        if (iter_idx + 1) % args.eval_freq == 0:
            torch.save(model.state_dict(), 'models/order_{}.pth'.format(iter_idx))
            rollout = einops.rearrange(attns, 'b f (h w) q -> b (f h) (q w)', w = image_size[-1]//image_patch_size).detach().cpu().numpy()
            writer.add_images('train_attn1', np.expand_dims(rollout[0], (0,1)), iter_idx+1, dataformats='NCHW')
            writer.add_images('train_attn2', np.expand_dims(rollout[1], (0,1)), iter_idx+1, dataformats='NCHW')
            writer.add_images('train_attn3', np.expand_dims(rollout[2], (0,1)), iter_idx+1, dataformats='NCHW')
            writer.add_images('train_attn4', np.expand_dims(rollout[3], (0,1)), iter_idx+1, dataformats='NCHW')

            data_ = data[:4,:,:3]/255.
            writer.add_images('train_input_seq1', einops.rearrange(data_[0], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            writer.add_images('train_input_seq2', einops.rearrange(data_[1], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            writer.add_images('train_input_seq3', einops.rearrange(data_[2], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            writer.add_images('train_input_seq4', einops.rearrange(data_[3], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')

            current_valid_accs = []

            for data, targets in data_loader_valid:
                data, targets = data.to(args.device), targets.to(args.device)
                dic, sam = utils.ranking_accuracy_mono(model, data, targets, args.reversible)
                current_valid_accs.append(dic)
                writer.add_images('to_sam', np.expand_dims(sam, (0,1)), iter_idx+1, dataformats='NCHW')

            valid_accs.append(utils.avg_list_of_dicts(current_valid_accs))

            print(iter_idx, 'valid', valid_accs[-1])
            print(sum(losses)/len(losses))

            data_ = data[:4,:,:3]/255.
            writer.add_images('input_seq1', einops.rearrange(data_[0], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            writer.add_images('input_seq2', einops.rearrange(data_[1], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            writer.add_images('input_seq3', einops.rearrange(data_[2], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            writer.add_images('input_seq4', einops.rearrange(data_[3], '(dn n) c h w -> dn c (n h) w', dn=1), iter_idx+1, dataformats='NCHW')
            
            with torch.no_grad():
                data, attns = model(data)

            rollout = einops.rearrange(attns, 'b f (h w) q -> b (f h) (q w)', w = image_size[-1]//image_patch_size).cpu().numpy()
            writer.add_images('attn1', np.expand_dims(rollout[0], (0,1)), iter_idx+1, dataformats='NCHW')
            writer.add_images('attn2', np.expand_dims(rollout[1], (0,1)), iter_idx+1, dataformats='NCHW')
            writer.add_images('attn3', np.expand_dims(rollout[2], (0,1)), iter_idx+1, dataformats='NCHW')
            writer.add_images('attn4', np.expand_dims(rollout[3], (0,1)), iter_idx+1, dataformats='NCHW')
            
            writer.add_scalar('val/acc_em5', valid_accs[-1]['acc_em5'], iter_idx+1)
            writer.add_scalar('val/acc_ew5', valid_accs[-1]['acc_ew5'], iter_idx+1)
            writer.add_scalar('val/acc_em', valid_accs[-1]['acc_em'], iter_idx+1)
            writer.add_scalar('val/acc_ew', valid_accs[-1]['acc_ew'], iter_idx+1)


            writer.add_scalar('trn/loss', sum(losses)/len(losses), iter_idx+1)
            writer.add_scalar('trn/acc_em', sum(train_ems)/len(train_ems), iter_idx+1)
            writer.add_scalar('trn/acc_ew', sum(train_ews)/len(train_ews), iter_idx+1)
            losses = []
            train_ems = []
            train_ews = []
