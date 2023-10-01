import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import get_dataset, DensityCollator, DensityVoxelCollator
from models import get_model
from utils import load_config, seed_all, get_optimizer, get_scheduler, count_parameters
from visualize import draw_stack

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InfGCN Training/Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train',
                        help='running mode: train or inf')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')
    parser.add_argument('--savename', type=str, default='test', help='save name')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume from')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Data
    print('Loading datasets...')
    use_voxel = config.model.type == 'cnn'
    if use_voxel:
        train_collator = val_collator = inf_collator = DensityVoxelCollator()
    else:
        train_collator = DensityCollator(config.train.train_samples)
        val_collator = DensityCollator(config.train.val_samples)
        inf_collator = DensityCollator()
    train_set, val_set, test_set = get_dataset(config.datasets)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True,
                              num_workers=32, collate_fn=train_collator)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            num_workers=32, collate_fn=val_collator)
    inf_loader = DataLoader(val_set, 2, shuffle=True, num_workers=2, collate_fn=inf_collator)

    # Model
    print('Building model...')
    model = get_model(config.model).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    criterion = nn.MSELoss().to(args.device)
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])

    global_step = 0


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for g, density, grid_coord, infos in train_loader:
                g = g.to(args.device)
                density, grid_coord = density.to(args.device), grid_coord.to(args.device)
                pred = model(g.x, g.pos, grid_coord, g.batch, infos)
                if use_voxel:
                    mask = (density > 0).float()
                    pred = pred * mask
                    density = density * mask
                loss = criterion(pred, density)
                mae = torch.abs(pred.detach() - density).sum() / density.sum()
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/mae', mae.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f},'
                          f' train mae {mae.item():.6f}')
                global_step += 1
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate(val_loader)
                    inference(inf_loader, 1, config.test.num_vis, config.test.inf_samples)

                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    torch.save({
                        'model': model.state_dict(),
                        'step': global_step,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate(dataloader, split='val'):
        with torch.no_grad():
            model.eval()

            val_losses = []
            val_mae, val_cnt = 0., 0.
            for g, density, grid_coord, infos in tqdm(dataloader, total=len(dataloader)):
                g = g.to(args.device)
                density, grid_coord = density.to(args.device), grid_coord.to(args.device)
                pred = model(g.x, g.pos, grid_coord, g.batch, infos)
                if use_voxel:
                    mask = (density > 0).float()
                    pred = pred * mask
                    density = density * mask
                loss = criterion(pred, density)
                val_losses.append(loss.item())
                val_mae += torch.abs(pred - density).sum().item()
                val_cnt += density.sum().item()
        val_loss = sum(val_losses) / len(val_losses)
        val_mae = val_mae / val_cnt

        writer.add_scalar(f'{split}/loss', val_loss, global_step)
        writer.add_scalar(f'{split}/mae', val_mae, global_step)
        print(f'Step {global_step} {split} loss {val_loss:.6f}, {split} mae {val_mae:.6f}')
        return val_loss


    def inference_batch(g, density, grid_coord, infos, grid_batch_size=None):
        with torch.no_grad():
            model.eval()
            if grid_batch_size is None:
                preds = model(g.x, g.pos, grid_coord, g.batch, infos)
            else:
                preds = []
                for grid in grid_coord.split(grid_batch_size, dim=1):
                    preds.append(model(g.x, g.pos, grid.contiguous(), g.batch, infos))
                preds = torch.cat(preds, dim=1)
            mask = (density > 0).float()
            preds = preds * mask
            density = density * mask
            diff = torch.abs(preds - density)
            sum_idx = tuple(range(1, density.dim()))
            loss = diff.pow(2).sum(sum_idx) / mask.sum(sum_idx)
            mae = diff.sum(sum_idx) / density.sum(sum_idx)
        return preds, loss, mae


    def inference(dataloader, num_infer=None, num_vis=2, samples=None):
        inf_loss, inf_mae = [], []
        num_infer = num_infer or len(dataloader)
        for idx, (g, density, grid_coord, infos) in tqdm(enumerate(dataloader), total=num_infer):
            if idx >= num_infer:
                break

            g = g.to(args.device)
            density, grid_coord = density.to(args.device), grid_coord.to(args.device)
            pred, loss, mae = inference_batch(g, density, grid_coord, infos, samples)
            inf_loss.append(loss.detach().cpu().numpy())
            inf_mae.append(mae.detach().cpu().numpy())

            if idx == 0:
                for vis_idx, (p, d, info) in enumerate(zip(pred, density, infos)):
                    if vis_idx >= num_vis:
                        break

                    shape = info['shape']
                    mask = g.batch == vis_idx
                    atom_type, coord = g.x[mask], g.pos[mask]
                    grid_cell = (info['cell'] / torch.FloatTensor(shape).view(3, 1)).to(args.device)
                    coord = coord @ torch.linalg.inv(grid_cell)
                    if use_voxel:
                        d = d[:shape[0], :shape[1], :shape[2]]
                        p = p[:shape[0], :shape[1], :shape[2]]
                    else:
                        num_voxel = shape[0] * shape[1] * shape[2]
                        d, p = d[:num_voxel].view(*shape), p[:num_voxel].view(*shape)
                    writer.add_image(f'inf/gt_{vis_idx}', draw_stack(d, atom_type, coord), global_step)
                    writer.add_image(f'inf/pred_{vis_idx}', draw_stack(p, atom_type, coord), global_step)
                    writer.add_image(f'inf/diff_{vis_idx}', draw_stack(d - p, atom_type, coord), global_step)
        inf_loss = np.concatenate(inf_loss, axis=0).mean()
        inf_mae = np.concatenate(inf_mae, axis=0).mean()
        writer.add_scalar('inf/loss', inf_loss, global_step)
        writer.add_scalar('inf/mae', inf_mae, global_step)
        print(f'Step {global_step} inference loss {inf_loss:.6f}, inference mae {inf_mae:.6f}')


    try:
        if args.mode == 'train':
            # inference(inf_loader, 1, config.test.num_vis, config.test.inf_samples)
            train()
            print('Training finished!')

        if args.mode == 'inf' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')
        test_loader = DataLoader(test_set, config.test.batch_size, shuffle=False,
                                 num_workers=16, collate_fn=inf_collator)
        inference(test_loader, config.test.num_infer, config.test.num_vis, config.test.inf_samples)
    except KeyboardInterrupt:
        print('Terminating...')
