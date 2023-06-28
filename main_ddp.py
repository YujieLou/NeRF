import os
import logging
import time
import imageio
import numpy as np
from tqdm import tqdm, trange
import configargparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *
from models import NeRF
from dataloader import load_llff_data

seed_everything(0)

def get_logger(filename, distributed_rank=0, verbosity=1, name=None):
    if distributed_rank > 0:
        logging_not_root = logging.getLogger(name=__name__)
        logging_not_root.propagate = False
        return logging_not_root
    
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger



def render(H, W, K, args, rays=None, near=0., far=1., 
           c2w_staticcam=None, model=None):
    """Render rays
    Args:
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays.
    """
    rays_o, rays_d = rays
        
    if args.use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if not args.no_ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if args.use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = {}
    for i in range(0, rays.shape[0], args.chunk):
        ret = model(rays[i:i+args.chunk])
    
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def test(render_poses, hwf, K, model, args, near, far, savedir=None, render_factor=0, logger=None):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs, disps = [], []

    for i, c2w in enumerate(tqdm(render_poses)):
        t = time.time()
        rays = get_rays(H, W, K, c2w[:3,:4])
        rgb, disp, acc, _ = render(H, W, K, rays = rays, 
                                   model=model, 
                                   args=args, near=near, far=far,)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(rgbs[-1]))
        logger.info('[Test] Image id :{}  Time consume : {}'.format(i, time.time() - t))
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps



def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('config', is_config_file=True,   help='config file path')
    parser.add_argument("--expname", type=str,  help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')

    # training options
    parser.add_argument("--encode_mode", type=str, default='PE', help='encoding mode, options: PE / Hash')
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--epochs",   type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32*32, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--checkpoint", type=str, default=None, help='specific weights npy file to reload for coarse network')
    
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--use_viewdirs", action='store_true',  help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')
    
    parser.add_argument("--remove_car", action='store_true',  help='remove cars in images')
    parser.add_argument("--render_only", action='store_true',  help='only test')
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser



def main(args, logger):
    ###############
    # Load data
    ###############
    if args.dataset_type == 'llff':
        images, hwf, poses, bds, render_poses, i_test, car_bounds = load_llff_data(args, recenter=True, bd_factor=.75)
        logger.info('Loaded llff {} {} {} {}'.format(images.shape, render_poses.shape, hwf, args.datadir))

        if not isinstance(i_test, list):
            i_test = [i_test]
  
        if args.llffhold > 0:
            logger.info('Auto LLFF holdout, {}'.format(args.llffhold))
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test)])
        # i_test = np.arange(images.shape[0])
        
        logger.info('DEFINING BOUNDS')
        if not args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.    
        else:
            near = 0.
            far = 1.

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([[focal, 0, 0.5*W],[0, focal, 0.5*H],[0, 0, 1]])

    # Prepare raybatch tensor if batching random rays
    logger.info('get rays')
    if args.remove_car:
        rays = np.stack([get_rays_np_choice(H, W, K, poses[i,:3,:4], car_bounds[i]) for i in range(len(poses))], 0) # [N, ro+rd, H, W, 3]
    else:
        rays = np.stack([get_rays(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    logger.info('done, concats')
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)

    if args.remove_car:
        rays_rgb = remove_car(rays_rgb)

    logger.info('done')
    # Move data to GPU
    rays_rgb = torch.Tensor(rays_rgb).to(args.gpu)  # [(N-1)*H*W, ro+rd+rgb, 3]
    images = torch.Tensor(images).to(args.gpu)
    poses = torch.Tensor(poses).to(args.gpu)
    render_poses = torch.Tensor(render_poses).to(args.gpu)

    train_sampler = torch.utils.data.distributed.DistributedSampler(rays_rgb)
    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(rays_rgb, batch_size=args.batch_size, sampler=train_sampler)


    #####################
    # Create nerf model
    #####################
    model = NeRF(args)

        
    # Create optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    #####################
    # Load checkpoints
    #####################
    start = 0
    if args.checkpoint is not None and args.checkpoint!='None':
        ckpts = [args.checkpoint]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if 'tar' in f]

    logger.info('Found ckpts {}'.format(ckpts))
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        logger.info('Reloading from {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['model_state_dict'])

    model = DDP(model, device_ids=[args.local_rank],output_device=args.local_rank)



    if args.render_only:
        dir = os.path.join(args.basedir, args.expname, 'render_only_{:06d}'.format(start))
        os.makedirs(dir, exist_ok=True)
        with torch.no_grad():
            rgbs, disps = test(render_poses, hwf, K, 
                               model, 
                               args, near, far, savedir=dir, logger=logger)
        return 

    #####################
    # Train
    #####################
    logger.info('Begin')
    logger.info('TRAIN views are {}'.format(i_train))
    logger.info('TEST views are {}'.format(i_test))
    
    global_step = start
    epochs = args.epochs+1
    start = start + 1
    # i_batch = 0

    for epoch in trange(start, epochs):
        trainloader.sampler.set_epoch(epoch)
        for batch in trainloader:
            batch = batch.permute(1, 0, 2)
            batch_rays, target_s = batch[:2], batch[2]

            #####  Core optimization loop  #####
            with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
                rgb, disp, acc, extras = render(H, W, K, args, rays=batch_rays, 
                                                        model=model,
                                                        near=near, far=far)
            # print(prof.table())
            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
            loss.backward()
            optimizer.step()

            
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            # Rest is logging
        if epoch%args.i_weights==0:
            path = os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(epoch))
            torch.save({
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            logger.info('Saved checkpoints at {}'.format(path))

        logger.info('New Learning Rate {}'.format(new_lrate))
        logger.info(f"[TRAIN] Epoch: {epoch}/{epochs-start}  Loss: {loss.item()}  PSNR: {psnr.item()}.")
        
        if epoch%args.i_testset==0 and epoch > 0:
            testsavedir = os.path.join(args.basedir, args.expname, 'testset_{:06d}'.format(epoch))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info('test poses shape {}'.format(poses[i_test].shape))
            with torch.no_grad():
                model.status = 'test'
                rgbs, disps = test(poses[i_test], hwf, K, model, 
                                   args, near, far, savedir=testsavedir, logger=logger)
                model.status = 'train'
            
            moviebase = os.path.join(args.basedir, args.expname, '{}_spiral_{:06d}_'.format(args.expname, epoch))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            logger.info('Saved test set')
            
        global_step += 1



if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, 'args.txt')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    args.local_rank = int(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.devices = [args.local_rank]
    args.gpu = torch.device(f'cuda:{args.local_rank:d}')
    
    logger = get_logger(os.path.join(args.basedir, args.expname, 'log.log'), args.local_rank)
    logger.info('Device ID: {}'.format(args.devices))
    main(args, logger)
