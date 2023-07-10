import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn
import time
import copy
import numpy as np 


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply

def contract(x: torch.Tensor, radius: float=1.0):
    x_norm = x.norm(dim=1, keepdim=True)
    msk = x_norm<=radius
    return x*msk + ~msk * (1 + radius - radius / x_norm) * x / x_norm


def volume_rendering(raw, z_vals, rays_d, raw_noise_std=0, device=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).to(device) * raw_noise_std
    # kkk = copy.deepcopy(dists)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = weights.add(0.0000001)  # 解决办法

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))  # nan的原因的weight有0

    '''
    if torch.isnan(disp_map).any():
        torch.set_printoptions(profile='full')
        for id, item in enumerate(disp_map):
            if torch.isnan(item):
                idx = id
        midd = raw[...,3]
        # print('alpha', alpha.shape, alpha[idx], '\n')
        print('raw[...,3]', midd.shape)
        print('kkk', kkk.shape, kkk[idx], '\n')
        # print('z_vals', z_vals.shape, z_vals[idx])
        # print('acc_map', acc_map.shape, acc_map[idx], '\n')
        torch.set_printoptions(profile='default')
    '''
    return rgb_map, disp_map, acc_map, weights, depth_map


def volume_rendering_ngp(raw, z_vals, rays_d, density, raw_noise_std=0, device=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    rgb = torch.sigmoid(raw)  # [N_rays, N_samples, 3]
    density = torch.reshape(density, [raw.shape[0], raw.shape[1]])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(density.shape).to(device) * raw_noise_std
    alpha = raw2alpha(density + noise, dists)  # [N_rays, N_samples]
    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = weights.add(0.0000001)  # 解决办法
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))  # nan的原因的weight有0

    return rgb_map, disp_map, acc_map, weights, depth_map



# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False, device=None):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, include_input, multires, log_sampling, periodic_fns):
        
        embed_fns = []
        d = 3
        out_dim = 0
        if include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = multires-1
        N_freqs = multires
        
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# Model
class NeRFMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRFMLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    


class NeRF(nn.Module):
    def __init__(self, args):
        super(NeRF, self).__init__()
        self.args = args
        '''
        self.embed = tcnn.Encoding(n_input_dims=3, encoding_config={
            'otype':'Frequency',
            'n_level':16,
            'n_features_per_level':2,
            'log2_hashmap_size':15,
            'base_resolution':16,
            'per_level_scale':1.5
        }).to(args.gpu)
        self.input_ch = 72
        '''
        embed_kwargs = {
            'include_input' : True,
            'multires' : 10,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],
            }
        embedder_obj = Embedder(**embed_kwargs)
        self.embed = lambda x, eo=embedder_obj:eo.embed(x)
        self.input_ch = embedder_obj.out_dim

            
        self.embeddir = None
        if args.use_viewdirs:
            '''
            self.embeddir = tcnn.Encoding(n_input_dims=3, encoding_config={
                'otype':'Frequency',
                'n_level':16,
                'n_features_per_level':2,
                'log2_hashmap_size':5,
                'base_resolution':16,
                'per_level_scale':1.5
            }).to(args.gpu)
            self.input_ch_views = 72
            '''
            embeddir_kwargs = {
                'include_input' : True,
                'multires' : 4,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                }
            embedderdir_obj = Embedder(**embeddir_kwargs)
            self.embeddir = lambda x, eo=embedderdir_obj:eo.embed(x)
            self.input_ch_views = embedderdir_obj.out_dim


                
        self.model = NeRFMLP(D=8, W=256,
                            input_ch=self.input_ch, output_ch=5 if args.N_importance > 0 else 4, skips=[4],
                            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs).cuda()


        self.model_fine = None
        if args.N_importance > 0:
            self.model_fine = NeRFMLP(D=8, W=256,
                                input_ch=self.input_ch, output_ch=5 if args.N_importance > 0 else 4, skips=[4],
                                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs).cuda()

        self.status = 'train'


    def forward(self, ray_batch):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3]
        directions = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=self.args.N_samples).to(self.args.gpu)
        if not self.args.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.args.N_samples])
        
        if self.status == 'train':
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(self.args.gpu)
            z_vals = lower + (upper - lower) * t_rand

  
        inputs_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        inputs_pts_flat = torch.reshape(inputs_pts, [-1, inputs_pts.shape[-1]])
        if self.args.contract:
            inputs_pts_flat = contract(inputs_pts_flat)


        # 65536, 3 --->  65536, 63
        embedded = self.embed(inputs_pts_flat)
        if directions is not None:
            input_dirs = directions[:,None].expand(inputs_pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 65536, 3 --->  65536, 27
        embedded_dirs = self.embeddir(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        # embedded = embedded.to(dtype=torch.float32)
        outputs_flat = self.model(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs_pts.shape[:-1]) + [outputs_flat.shape[-1]])   # batch_size, N_samples, 4
        rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering(outputs, z_vals, rays_d, 
                                                                          self.args.raw_noise_std if (self.status=='train') else 0., 
                                                                          self.args.gpu)
        # if (torch.isnan(disp_map).any() or torch.isinf(disp_map).any()):
            # torch.set_printoptions(profile='full')
            # print('disp_map 首次出现Nan')
            # print('disp_map', disp_map)
            # print('disp_map', disp_map.shape, 'max', max(disp_map),'min', min(disp_map))
            # print('exam outputs', torch.isnan(outputs).any())
            # print('exam z_vals', torch.isnan(z_vals).any())
            # print('exam rays_d', torch.isnan(rays_d).any())
            # time.sleep(1000000)
            # torch.set_printoptions(profile='default')

            
        if self.args.N_importance > 0 :
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.args.N_importance, 
                                   det=(self.status=='train'), device=self.args.gpu)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            inputs_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            inputs_pts_flat = torch.reshape(inputs_pts, [-1, inputs_pts.shape[-1]])
            if self.args.contract:
                inputs_pts_flat = contract(inputs_pts_flat)
            # 65536, 3 --->  65536, 63
            embedded = self.embed(inputs_pts_flat)
            if directions is not None:
                input_dirs = directions[:,None].expand(inputs_pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            # 65536, 3 --->  65536, 27
            embedded_dirs = self.embeddir(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)
            embedded = embedded.to(dtype=torch.float32)
            outputs_flat = self.model_fine(embedded)
            outputs = torch.reshape(outputs_flat, list(inputs_pts.shape[:-1]) + [outputs_flat.shape[-1]])

            rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering(outputs, z_vals, rays_d, 
                                                                              self.args.raw_noise_std if (self.status=='train') else 0., 
                                                                              self.args.gpu)
        
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

        if self.status=='train': 
            ret['raw'] = outputs
        if self.args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")
        return ret


class NGP(nn.Module):
    def __init__(self, args):
        super(NGP, self).__init__()
        self.args = args

        self.n_levels = 16 # 16  调高能够提高远近物体的分离度  效果很小很小
        self.n_features_per_level = 2
        self.log2_hashmap_size = 19   # 哈希表的尺寸？  严重影响运行速度
        self.max_resolution = 4096
        self.base_resolution = 16
        self.per_level_scale = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1)
        ).tolist()
        self.embed = tcnn.Encoding(n_input_dims=3, encoding_config={
                'otype':'HashGrid',
                'n_level':self.n_levels,
                'n_features_per_level':self.n_features_per_level,  ##
                'log2_hashmap_size':self.log2_hashmap_size,  
                'base_resolution':self.base_resolution, 
                'per_level_scale': self.per_level_scale
            }).to(args.gpu)
        self.input_ch = self.n_features_per_level * 16  # n_features_per_level * 16

        
        self.embeddir = None
        if args.use_viewdirs:
            self.embeddir = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'Composite',
                    'nested': [
                        {
                            'n_dims_to_encode':3,
                            'otype': 'SphericalHarmonics',
                            'degree': 4,
                        },
                        # {'otype': 'Identity', 'n_bins': 4, 'degree': 4}
                    ]                        
                }                 
            ).to(args.gpu)
            self.input_ch_views = 16

        # D=2, W=64
        # D=4, W=128
        # D=8, W=256
        self.model = NeRFMLP(D=2, W=64,
                            input_ch=self.input_ch, output_ch=5 if args.N_importance > 0 else 4, skips=[4],
                            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs).cuda()

        self.model_fine = None
        if args.N_importance > 0:
            self.model_fine = NeRFMLP(D=2, W=64,
                                input_ch=self.input_ch, output_ch=5 if args.N_importance > 0 else 4, skips=[4],
                                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs).cuda()
        self.status = 'train'

    def forward(self, ray_batch):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3]
        directions = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=self.args.N_samples).to(self.args.gpu)
        if not self.args.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.args.N_samples])
        
        if self.status == 'train':
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(self.args.gpu)
            z_vals = lower + (upper - lower) * t_rand

  
        inputs_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        inputs_pts_flat = torch.reshape(inputs_pts, [-1, inputs_pts.shape[-1]])
        # 65536, 3 --->  65536, 63
        if self.args.contract:
            inputs_pts_flat = contract(inputs_pts_flat)
        embedded = self.embed(inputs_pts_flat)
        if directions is not None:
            input_dirs = directions[:,None].expand(inputs_pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 65536, 3 --->  65536, 27
        embedded_dirs = self.embeddir(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        embedded = embedded.to(dtype=torch.float32)
        outputs_flat = self.model(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs_pts.shape[:-1]) + [outputs_flat.shape[-1]])   # batch_size, N_samples, 4
        rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering(outputs, z_vals, rays_d, 
                                                                          self.args.raw_noise_std if (self.status=='train') else 0., 
                                                                          self.args.gpu)

            
        if self.args.N_importance > 0 :
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.args.N_importance, 
                                   det=(self.status=='train'), device=self.args.gpu)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            inputs_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            inputs_pts_flat = torch.reshape(inputs_pts, [-1, inputs_pts.shape[-1]])
            if self.args.contract:
                inputs_pts_flat = contract(inputs_pts_flat)
            # 65536, 3 --->  65536, 63
            embedded = self.embed(inputs_pts_flat)
            if directions is not None:
                input_dirs = directions[:,None].expand(inputs_pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            # 65536, 3 --->  65536, 27
            embedded_dirs = self.embeddir(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)
            embedded = embedded.to(dtype=torch.float32)
            outputs_flat = self.model_fine(embedded)
            outputs = torch.reshape(outputs_flat, list(inputs_pts.shape[:-1]) + [outputs_flat.shape[-1]])

            rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering(outputs, z_vals, rays_d, 
                                                                              self.args.raw_noise_std if (self.status=='train') else 0., 
                                                                              self.args.gpu)
        
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

        if self.status=='train': 
            ret['raw'] = outputs
        if self.args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")
        return ret



