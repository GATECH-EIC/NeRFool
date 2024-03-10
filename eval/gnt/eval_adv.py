import os
import sys

import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import torch.nn as nn

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_image import render_single_image
from gnt.model import GNTModel
from gnt.sample_ray import RaySamplerSingleImage, rng
from gnt.render_ray import render_rays
from gnt.criterion import Criterion
from utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
import config
import torch.distributed as dist
from gnt.projection import Projector
from gnt.data_loaders.create_training_dataset import create_training_dataset
import imageio

from train import calc_depth_var
from geo_interp import interp3
from pc_grad import PCGrad


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def calc_depth_smooth_loss(ret, patch_size, loss_type='l2'):
    depth = ret['depth']  # [N_rays,], i.e., [n_patches * patch_size * patch_size]

    depth = depth.reshape([-1, patch_size, patch_size])  # [n_patches, patch_size, patch_size]

    v00 = depth[:, :-1, :-1]
    v01 = depth[:, :-1, 1:]
    v10 = depth[:, 1:, :-1]

    if loss_type == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif loss_type == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported loss type.')

    return loss.sum()


class SL1Loss(nn.Module):
    def __init__(self):
        super(SL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None, useMask=True):
        if None == mask and useMask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask])  # * 2 ** (1 - 2)
        return loss


### modified from Sin-NeRF
def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src,
                       extrinsics_src):  # (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')
    '''
    depth_ref: [B, H, W]
    intrinsics_ref: [3, 3]
    extrinsics_ref: [4, 4]
    '''
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                   torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    pts = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(
        0) * (depth_ref.view(batchsize, -1).unsqueeze(1))  ## [B, 3, height * width]

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), pts)  ## [B, 3, height * width]

    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:,
              :3, :]  ## [B, 3, height * width]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480, [B, 3, height * width]
    depth_src = K_xyz_src[:, 2:3, :]  ## [B, 1, height * width]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-9)  ## [B, 2, height * width]
    x_src = xy_src[:, 0, :].view([batchsize, height, width])  ## [B, height, width]
    y_src = xy_src[:, 1, :].view([batchsize, height, width])  ## [B, height, width]

    return x_src, y_src, depth_src


def forward_warp(selected_inds, rgb_ref, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src,
                 src2tar=True, derive_full_image=False, cpu_speedup=True):
    '''
    selected_inds: [Num_Sampled_Rays] from the target view
    rgb_ref: [H, W, C]
    depth_ref: [B, H, W] or [H, W]
    intrinsics_ref: [3, 3]
    extrinsics_ref: [4, 4]
    cpu_speedup: Put the indexing operations to CPU for further speed-up
    '''

    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    if cpu_speedup:
        new = torch.zeros([height, width, 3])  ## pseudo RGB for the image, [height, width, 3]

        depth_src = depth_src.reshape(height, width).cpu()  ## [height, width], assume batch_size=1
        new_depth = torch.zeros_like(depth_src)  ## [height, width], depth for the whole source image
        x_res = x_res.cpu()
        y_res = y_res.cpu()
        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long), torch.arange(0, width, dtype=torch.long)])

    else:
        new = torch.zeros([height, width, 3]).to(depth_ref.device)  ## pseudo RGB for the image, [height, width, 3]

        depth_src = depth_src.reshape(height, width)  ## [height, width], assume batch_size=1
        new_depth = torch.zeros_like(depth_src)  ## [height, width], depth for the whole source image

        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long, device=depth_ref.device),
            torch.arange(0, width, dtype=torch.long, device=depth_ref.device)])

    y_res = torch.clamp(y_res, 0, height - 1).to(torch.long)  ## [B, height, width]
    x_res = torch.clamp(x_res, 0, width - 1).to(torch.long)  ## [B, height, width]
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)  ## [height*width], assume batch_size=1
    x_res = x_res.reshape(-1)  ## [height*width], assume batch_size=1

    if not derive_full_image:
        if src2tar:
            inds_res = (y_res * width + x_res).cpu().numpy()

            for i, inds in enumerate(inds_res):
                if inds in selected_inds:
                    if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[
                        yy_base[i], xx_base[i]]:
                        new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                        new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

            depth_proj = new_depth.reshape(-1)[selected_inds]  ## [N_rand]
            rgb_proj = new.reshape(-1, 3)[selected_inds]  ## [N_rand, 3]

            if cpu_speedup:
                new = new.to(depth_ref.device)
                new_depth = new_depth.to(depth_ref.device)
                rgb_proj = rgb_proj.to(depth_ref.device)
                depth_proj = depth_proj.to(depth_ref.device)

            return new, new_depth, rgb_proj, depth_proj

        else:
            selected_inds_new = []
            for i in selected_inds:
                selected_inds_new.append((y_res[i] * width + x_res[i]).item())
                if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[
                    yy_base[i], xx_base[i]]:
                    new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                    new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

            depth_proj = new_depth.reshape(-1)[selected_inds_new]  ## [N_rand]
            rgb_proj = new.reshape(-1, 3)[selected_inds_new]  ## [N_rand, 3]

            if cpu_speedup:
                new = new.to(depth_ref.device)
                new_depth = new_depth.to(depth_ref.device)
                rgb_proj = rgb_proj.to(depth_ref.device)
                depth_proj = depth_proj.to(depth_ref.device)

            return new, new_depth, rgb_proj, depth_proj, selected_inds_new

    else:
        # painter's algo
        for i in range(yy_base.shape[0]):
            if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

        depth_proj = new_depth.reshape(-1)[selected_inds]  ## [N_rand]
        rgb_proj = new.reshape(-1, 3)[selected_inds]  ## [N_rand, 3]

        if cpu_speedup:
            new = new.to(depth_ref.device)
            new_depth = new_depth.to(depth_ref.device)
            rgb_proj = rgb_proj.to(depth_ref.device)
            depth_proj = depth_proj.to(depth_ref.device)

        return new, new_depth, rgb_proj, depth_proj


def calc_rotation_matrix(rot_degree):
    '''
    input: rot_degree - [3]
    output: rotation matrix - [3, 3]
    '''

    dx, dy, dz = rot_degree

    rot_x = torch.zeros(3, 3)
    rot_x[0, 0] = torch.cos(dx)
    rot_x[0, 1] = -torch.sin(dx)
    rot_x[1, 0] = torch.sin(dx)
    rot_x[1, 1] = torch.cos(dx)
    rot_x[2, 2] = 1

    rot_y = torch.zeros(3, 3)
    rot_y[0, 0] = torch.cos(dy)
    rot_y[0, 2] = torch.sin(dy)
    rot_y[1, 1] = 1
    rot_y[2, 0] = -torch.sin(dy)
    rot_y[2, 2] = torch.cos(dy)

    rot_z = torch.zeros(3, 3)
    rot_z[0, 0] = 1
    rot_z[1, 1] = torch.cos(dz)
    rot_z[1, 2] = -torch.sin(dz)
    rot_z[2, 1] = torch.sin(dz)
    rot_z[2, 2] = torch.cos(dz)

    rot_mat = rot_z.mm(rot_y.mm(rot_x))  # shape=[3, 3]

    return rot_mat


def transform_src_cameras(src_cameras, rot_param, trans_param, num_source_views):
    camera_pose = src_cameras[0, :, -16:].reshape(-1, 4, 4)  # [num_source_views, 4, 4]

    rot_mats = []
    for src_id in range(num_source_views):
        rot_mat = calc_rotation_matrix(rot_param[src_id])  # [3, 3]
        rot_mats.append(rot_mat.unsqueeze(0))
    rot_mats = torch.cat(rot_mats, dim=0).to(src_cameras)  # [num_source_views, 3, 3]

    rot_new = rot_mats.bmm(camera_pose[:, :3, :3])  # [num_source_views, 3, 3]
    trans_new = camera_pose[:, :3, 3] + trans_param.to(src_cameras)  # [num_source_views, 3]
    rot_trans = torch.cat([rot_new, trans_new.unsqueeze(2)], dim=2)  # [num_source_views, 3, 4]

    return rot_trans


def init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit):
    delta = torch.zeros_like(src_ray_batch['src_rgbs'])  # ray_batch['src_rgbs'].shape=[1, N_views, H, W, 3]
    delta.uniform_(-epsilon, epsilon)
    delta.data = clamp(delta, lower_limit - src_ray_batch['src_rgbs'], upper_limit - src_ray_batch['src_rgbs'])
    delta.requires_grad = True

    return delta


### Derive adv perturbations using PGD on training data
def optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True, criterion=None):
    if args.gt_depth_path:
        load_gt_depth = True
    else:
        load_gt_depth = False

    ray_sampler_train = RaySamplerSingleImage(data, 'cuda:0', load_gt_depth=load_gt_depth)

    if args.use_patch_sampling:
        train_ray_batch = ray_sampler_train.random_patch_sample(args.N_rand, patch_size=args.patch_size)
    else:
        train_ray_batch = ray_sampler_train.random_sample(args.N_rand, sample_mode=args.sample_mode,
                                                          center_ratio=args.center_ratio)

    if args.use_pseudo_gt:
        with torch.no_grad():
            featmaps = model.feature_net(
                src_ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))  ## extract features from test src_rgbs

            ret_gt = render_rays(ray_batch=train_ray_batch,
                                 model=model,
                                 projector=projector,
                                 featmaps=featmaps,
                                 N_samples=args.N_samples,
                                 inv_uniform=args.inv_uniform,
                                 N_importance=args.N_importance,
                                 det=True,  # args.det,
                                 white_bkgd=args.white_bkgd,
                                 ret_alpha=args.ret_alpha,
                                 args=args,
                                 src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

            train_ray_batch['rgb'] = ret_gt['outputs_coarse']['rgb']
            train_ray_batch['depth'] = ret_gt['outputs_coarse']['depth']

    total_loss = {}

    featmaps = model.feature_net(
        (src_ray_batch['src_rgbs'] + delta).squeeze(0).permute(0, 3, 1, 2))  ## extract features from test src_rgbs

    ret = render_rays(ray_batch=train_ray_batch,
                      model=model,
                      projector=projector,
                      featmaps=featmaps,
                      N_samples=args.N_samples,
                      inv_uniform=args.inv_uniform,
                      N_importance=args.N_importance,
                      det=args.det,
                      white_bkgd=args.white_bkgd,
                      ret_alpha=args.ret_alpha,
                      args=args,
                      src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

    loss_rgb, _ = criterion(ret['outputs_coarse'], train_ray_batch, scalars_to_log=None)
    if ret['outputs_fine'] is not None:
        fine_loss, _ = criterion(ret['outputs_fine'], train_ray_batch, scalars_to_log=None)
        loss_rgb = loss_rgb + fine_loss
    total_loss['rgb'] = loss_rgb

    if args.density_loss > 0:
        assert args.use_pseudo_gt
        from utils import img2mse
        loss_density = img2mse(ret['outputs_coarse']['alpha'], ret_gt['outputs_coarse']['alpha'])

        if ret['outputs_fine'] is not None:
            loss_density = loss_density + img2mse(ret['outputs_fine']['alpha'], ret_gt['outputs_fine']['alpha'])

        loss_density = args.density_loss * loss_density
        total_loss['density'] = loss_density

    if args.depth_var_loss > 0:
        depth_var = calc_depth_var(ret['outputs_coarse'])

        if ret['outputs_fine'] is not None:
            depth_var = depth_var + calc_depth_var(ret['outputs_fine'])

        loss_depth_var = args.depth_var_loss * depth_var
        total_loss['depth_var'] = loss_depth_var

    if args.depth_diff_loss > 0:
        sl1_loss = SL1Loss()
        depth_gt = train_ray_batch['depth']
        depth_pred = ret['outputs_coarse']['depth']

        depth_diff = sl1_loss(depth_pred, depth_gt, useMask=True)
        if ret['outputs_fine'] is not None:
            depth_pred = ret['outputs_fine']['depth']
            depth_diff = depth_diff + sl1_loss(depth_pred, depth_gt, useMask=True)

        loss_depth_diff = args.depth_diff_loss * depth_diff
        total_loss['depth_diff'] = loss_depth_diff

    ### add multi-view depth consistency loss following Sin-NeRF
    if args.depth_consistency_loss > 0:
        sl1_loss = SL1Loss()

        if args.ds_rgb:  # downsample rgb images to match the resolution of depth; otherwise use upsampled depth
            ray_sampler_train_cons = RaySamplerSingleImage(data, 'cuda:0', resize_factor=0.5,
                                                           load_gt_depth=True)  # to match the resolution of ground-truth depth

            if args.use_patch_sampling:
                train_ray_batch_cons = ray_sampler_train_cons.random_patch_sample(args.N_rand,
                                                                                  patch_size=args.patch_size)
            else:
                train_ray_batch_cons = ray_sampler_train_cons.random_sample(args.N_rand, sample_mode=args.sample_mode,
                                                                            center_ratio=args.center_ratio)

            featmaps = model.feature_net((src_ray_batch['src_rgbs'] + delta).squeeze(0).permute(0, 3, 1,
                                                                                                2))  ## extract features from test src_rgbs

            ret_cons = render_rays(ray_batch=train_ray_batch_cons,
                                   model=model,
                                   projector=projector,
                                   featmaps=featmaps,
                                   N_samples=args.N_samples,
                                   inv_uniform=args.inv_uniform,
                                   N_importance=args.N_importance,
                                   det=args.det,
                                   white_bkgd=args.white_bkgd,
                                   ret_alpha=args.ret_alpha,
                                   args=args,
                                   src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

        else:
            train_ray_batch_cons = train_ray_batch
            ret_cons = ret

        train_camera = train_ray_batch_cons['camera']  # [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        extrinsics_src = train_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_src = train_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_src = intrinsics_src[:3, :3]  # [3, 3]

        src_id = rng.choice(args.num_source_views)
        src_cameras = src_ray_batch['src_cameras']  # [1, num_source_views, 34]
        src_camera = src_cameras[0, src_id:src_id + 1, :]

        extrinsics_ref = src_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_ref = src_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_ref = intrinsics_ref[:3, :3]  # [3, 3]

        src_rgbs = src_ray_batch['src_rgbs']  # [1, num_source_views, H, W, 3]
        rgb_ref = src_rgbs[:, src_id, :, :]  # [1, H, W, 3]

        if args.ds_rgb:
            ### resize the rgb and intrinsics of the ref view to match the depth resolution
            resize_factor = 0.5
            intrinsics_ref[:2, :3] *= resize_factor
            rgb_ref = torch.nn.functional.interpolate(rgb_ref.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(
                0, 2, 3, 1)

        rgb_ref = rgb_ref[0]  # [H, W, 3]

        src_depths = src_ray_batch['src_depths']  # [1, num_source_views, H, W]
        depth_ref = src_depths[0, src_id:src_id + 1, :, :]  # [1, H, W]

        rgb_proj_full, depth_proj_full, rgb_proj, depth_proj = forward_warp(train_ray_batch_cons['selected_inds'],
                                                                            rgb_ref, depth_ref, intrinsics_ref,
                                                                            extrinsics_ref, intrinsics_src,
                                                                            extrinsics_src, src2tar=True,
                                                                            derive_full_image=False)

        depth_pred = ret_cons['outputs_coarse']['depth']
        loss_depth_cons = sl1_loss(depth_pred, depth_proj, useMask=True)
        if ret_cons['outputs_fine'] is not None:
            depth_pred = ret_cons['outputs_fine']['depth']
            loss_depth_cons = loss_depth_cons + sl1_loss(depth_pred, depth_proj, useMask=True)

        loss_depth_cons = args.depth_consistency_loss * loss_depth_cons
        total_loss['depth_cons'] = loss_depth_cons

    ### apply depth smooth loss following RegNeRF
    if args.depth_smooth_loss > 0:
        if not args.use_patch_sampling:
            train_ray_batch = ray_sampler_train.random_patch_sample(args.N_rand, patch_size=args.patch_size)

            featmaps = model.feature_net((src_ray_batch['src_rgbs'] + delta).squeeze(0).permute(0, 3, 1,
                                                                                                2))  ## extract features from test src_rgbs

            ret_smooth = render_rays(ray_batch=train_ray_batch,
                                     model=model,
                                     projector=projector,
                                     featmaps=featmaps,
                                     N_samples=args.N_samples,
                                     inv_uniform=args.inv_uniform,
                                     N_importance=args.N_importance,
                                     det=args.det,
                                     white_bkgd=args.white_bkgd,
                                     ret_alpha=args.ret_alpha,
                                     args=args,
                                     src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

        else:
            ret_smooth = ret

        loss_depth_smooth = calc_depth_smooth_loss(ret_smooth['outputs_coarse'], patch_size=args.patch_size)
        if ret_smooth['outputs_fine'] is not None:
            loss_depth_smooth = loss_depth_smooth + calc_depth_smooth_loss(ret_smooth['outputs_fine'],
                                                                           patch_size=args.patch_size)

        loss_depth_smooth = args.depth_smooth_loss * loss_depth_smooth
        total_loss['depth_smooth'] = loss_depth_smooth

    ### add multi-view consistency loss for perturbing the camera
    if args.camera_consistency_loss > 0:
        sl1_loss = SL1Loss()

        train_camera = train_ray_batch['camera']  # [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        extrinsics_tar = train_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_tar = train_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_tar = intrinsics_tar[:3, :3]  # [3, 3]

        src_id = rng.choice(args.num_source_views)
        src_cameras = src_ray_batch['src_cameras']  # [1, num_source_views, 34]
        src_camera = src_cameras[0, src_id:src_id + 1, :]

        depth_tar = train_ray_batch['depth_full']  # [1, H, W]
        rgb_tar = ray_sampler_train.rgb.reshape(ray_sampler_train.H, ray_sampler_train.W, 3)  # [H, W, 3]

        extrinsics_src = src_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_src = src_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_src = intrinsics_src[:3, :3]  # [3, 3]

        src_rgbs = src_ray_batch['src_rgbs']  # [1, num_source_views, H, W, 3]
        rgb_src = src_rgbs[:, src_id, :, :]  # [1, H, W, 3]
        rgb_src = rgb_src[0]  # [H, W, 3]

        src_depths = src_ray_batch['src_depths']  # [1, num_source_views, H, W]
        depth_src = src_depths[0, src_id:src_id + 1, :, :]  # [1, H, W]

        rgb_src2tar_full, depth_src2tar_full, rgb_src2tar, depth_src2tar = forward_warp(
            train_ray_batch['selected_inds'], rgb_src, depth_src, intrinsics_src, extrinsics_src, intrinsics_tar,
            extrinsics_tar, src2tar=True, derive_full_image=False)
        rgb_tar2src_full, depth_tar2src_full, rgb_tar2src, depth_tar2src, selected_inds_src = forward_warp(
            train_ray_batch['selected_inds'], rgb_tar, depth_tar, intrinsics_tar, extrinsics_tar, intrinsics_src,
            extrinsics_src, src2tar=False, derive_full_image=False)

        if args.perturb_camera_no_detach:
            rgb_tar_sampled = ret['outputs_coarse']['rgb']  # [N_rand, 3]
        else:
            rgb_tar_sampled = ret['outputs_coarse']['rgb'].detach()  # [N_rand, 3]
        depth_tar_sampled = train_ray_batch['depth']  # [N_rand]
        rgb_src_sampled = rgb_src.reshape(-1, 3)[selected_inds_src]  # [N_rand, 3]
        depth_src_sampled = depth_src.reshape(-1)[selected_inds_src]  # [N_rand]

        loss_camera_cons = args.cam_src2tar * sl1_loss(rgb_tar_sampled, rgb_src2tar,
                                                       useMask=True) + args.cam_tar2src * sl1_loss(rgb_src_sampled,
                                                                                                   rgb_tar2src,
                                                                                                   useMask=True)
        loss_camera_cons = loss_camera_cons + args.cam_depth * (
                    sl1_loss(depth_tar_sampled, depth_src2tar, useMask=True) + sl1_loss(depth_src_sampled,
                                                                                        depth_tar2src, useMask=True))

        loss_camera_cons = args.camera_consistency_loss * loss_camera_cons
        total_loss['camera_cons'] = loss_camera_cons

    loss = sum(total_loss.values())

    if return_loss:
        return loss, total_loss

    grad = torch.autograd.grad(loss, delta)[0].detach()

    return grad


def optimize_purif(args, purif, delta, model, projector, src_ray_batch, data, self_purification):
    assert args.gt_depth_path
    if args.gt_depth_path:
        load_gt_depth = True
    else:
        load_gt_depth = False

    if self_purification:
        src_id = rng.choice(args.num_source_views)

        src_cameras = src_ray_batch['src_cameras']  # [1, num_source_views, 34]
        src_camera = src_cameras[:, src_id, :]  # [1, 34]
        src_rgbs = src_ray_batch['src_rgbs'] + delta.detach()  # [1, num_source_views, H, W, 3]
        src_rgb = src_rgbs[:, src_id, :, :, :]  # [1, H, W, 3]

        new_data = {}
        for key in data:
            new_data[key] = data[key]
        new_data['rgb'] = src_rgb.cpu()
        new_data['camera'] = src_camera.cpu()

        if 'depth' in data:
            src_depths = src_ray_batch['src_depths']  # [1, num_source_views, H, W]
            src_depth = src_depths[:, src_id, :, :]  # [1, H, W]
            new_data['depth'] = src_depth.cpu()

        data = new_data

    ray_sampler_train = RaySamplerSingleImage(data, 'cuda:0', load_gt_depth=load_gt_depth)

    if args.use_patch_sampling:
        train_ray_batch = ray_sampler_train.random_patch_sample(args.N_rand, patch_size=args.patch_size)
    else:
        train_ray_batch = ray_sampler_train.random_sample(args.N_rand, sample_mode=args.sample_mode,
                                                          center_ratio=args.center_ratio)

    total_loss = {}

    featmaps = model.feature_net((src_ray_batch['src_rgbs'] + delta.detach() + purif).squeeze(0).permute(0, 3, 1, 2))

    ret = render_rays(ray_batch=train_ray_batch,
                      model=model,
                      projector=projector,
                      featmaps=featmaps,
                      N_samples=args.N_samples,
                      inv_uniform=args.inv_uniform,
                      N_importance=args.N_importance,
                      det=args.det,
                      white_bkgd=args.white_bkgd,
                      ret_alpha=args.ret_alpha,
                      args=args,
                      src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

    if self_purification:
        loss_rgb, _ = criterion(ret['outputs_coarse'], train_ray_batch, scalars_to_log=None)
        if ret['outputs_fine'] is not None:
            fine_loss, _ = criterion(ret['outputs_fine'], train_ray_batch, scalars_to_log=None)
            loss_rgb = loss_rgb + fine_loss
        total_loss['rgb'] = loss_rgb
    else:
        assert args.purif_consistency_loss > 0

    ### add multi-view consistency loss for perturbing the camera
    if args.purif_consistency_loss > 0:
        sl1_loss = SL1Loss()

        train_camera = train_ray_batch['camera']  # [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        extrinsics_tar = train_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_tar = train_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_tar = intrinsics_tar[:3, :3]  # [3, 3]

        src_id = rng.choice(args.num_source_views)
        src_cameras = src_ray_batch['src_cameras']  # [1, num_source_views, 34]
        src_camera = src_cameras[0, src_id:src_id + 1, :]

        depth_tar = train_ray_batch['depth_full']  # [1, H, W]
        rgb_tar = ray_sampler_train.rgb.reshape(ray_sampler_train.H, ray_sampler_train.W, 3)  # [H, W, 3]

        extrinsics_src = src_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_src = src_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_src = intrinsics_src[:3, :3]  # [3, 3]

        src_rgbs = src_ray_batch['src_rgbs'] + delta.detach()  # [1, num_source_views, H, W, 3]
        rgb_src = src_rgbs[:, src_id, :, :]  # [1, H, W, 3]
        rgb_src = rgb_src[0]  # [H, W, 3]

        src_depths = src_ray_batch['src_depths']  # [1, num_source_views, H, W]
        depth_src = src_depths[0, src_id:src_id + 1, :, :]  # [1, H, W]

        rgb_src2tar_full, depth_src2tar_full, rgb_src2tar, depth_src2tar = forward_warp(
            train_ray_batch['selected_inds'], rgb_src, depth_src, intrinsics_src, extrinsics_src, intrinsics_tar,
            extrinsics_tar, src2tar=True, derive_full_image=False)

        loss_camera_cons = sl1_loss(ret['outputs_coarse']['rgb'], rgb_src2tar, useMask=True)
        if ret['outputs_fine'] is not None:
            loss_camera_cons = loss_camera_cons + sl1_loss(ret['outputs_fine']['rgb'], rgb_src2tar, useMask=True)

        loss_camera_cons = args.purif_consistency_loss * loss_camera_cons
        total_loss['camera_cons'] = loss_camera_cons

    loss = sum(total_loss.values())

    return loss, total_loss


def eval(args):
    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    if args.run_val == False:
        print("-->Training Mode, loading train dataset!")
        # create training dataset
        dataset, sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=True if sampler is None else False,
        )
        iterator = iter(loader)
    else:
        print("-->Evaluation Mode, loading val dataset!")
        # create validation dataset
        dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
        loader = DataLoader(dataset, batch_size=1)
        iterator = iter(loader)

    if not args.no_attack and not args.view_specific:  # optimize generalizable adv perturb across different views
        print("==>optimize generalizable adv perturb across different views, loading training dataset ... ...")
        train_dataset = dataset_dict[args.eval_dataset](args, 'train', scenes=args.eval_scenes)
        train_loader = DataLoader(train_dataset, batch_size=1, worker_init_fn=lambda _: np.random.seed(),
                                  num_workers=args.workers,
                                  pin_memory=True, shuffle=True)
    else:
        print("==>training view-specific pertubation, don't need to load training dataset!")
        train_dataset = None
        train_loader = None

    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device)

    indx = 0
    psnr_scores = []
    lpips_scores = []
    ssim_scores = []
    delta = 0

    criterion = Criterion()

    if args.gt_depth_path:
        load_gt_depth = True
    else:
        load_gt_depth = False

    if not args.view_specific:
        for i, data in enumerate(loader):
            src_ray_sampler = RaySamplerSingleImage(data, device='cuda:0', load_gt_depth=load_gt_depth)
            src_ray_batch_glb = src_ray_sampler.get_all()  # global source views for all view directions
            break
    else:
        src_ray_batch_glb = None

    if args.no_attack:
        delta = 0
        print("No attack, Whether to Use View-specific Source Views:", args.view_specific)

    elif args.view_specific:
        print("Attack with View-specific Adv Perturbations")

    else:  # optimize generalizable adv perturb across different views
        print("Attack with Adv Perturbations Generalizable across Views...")

        src_ray_batch = src_ray_batch_glb

        epsilon = torch.tensor(args.epsilon / 255.).cuda()
        alpha = torch.tensor(args.adv_lr / 255.).cuda()
        upper_limit = 1
        lower_limit = 0

        if args.perturb_camera:
            rot_param = torch.zeros(args.num_source_views, 3).cuda()  ## 3 rotation degrees
            if not args.zero_camera_init:
                rot_param.uniform_(-args.rot_epsilon / 180 * np.pi, args.rot_epsilon / 180 * np.pi)
            rot_param.requires_grad = True

            trans_param = torch.zeros(args.num_source_views, 3).cuda()  ## 3 translation distances
            if not args.zero_camera_init:
                trans_param.uniform_(-args.trans_epsilon, args.trans_epsilon)
            trans_param.requires_grad = True

            src_cameras_orig = src_ray_batch['src_cameras'].clone()  # [1, num_source_views, 34]

        delta = init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit)

        if args.use_adam:
            if args.perturb_camera:
                params = [delta, rot_param, trans_param]
            else:
                params = [delta]

            opt = torch.optim.Adam(params, lr=args.adam_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)

            if args.use_pcgrad:
                opt = PCGrad(opt, num_source_views=args.num_source_views)

        print("Start adversarial perturbation...")

        continue_flag = True
        iters = 0
        while continue_flag:
            for data in train_loader:
                if args.use_unseen_views:
                    args.use_pseudo_gt = True  # as GT rgb/depth may not be available for unseen views

                    if args.sample_based_on_depth:
                        z_camera = np.array([pose[2, 2] for pose in train_dataset.render_poses])  # forward direction
                        p_camera = np.exp(z_camera / args.temp) / np.sum(np.exp(z_camera / args.temp))
                        poses_sample_id = np.random.choice(len(train_dataset.render_poses), size=3, p=p_camera,
                                                           replace=False)

                    else:
                        poses_sample_id = np.random.choice(len(train_dataset.render_poses), size=3, replace=False)

                    pose1 = train_dataset.render_poses[poses_sample_id[0]]
                    pose2 = train_dataset.render_poses[poses_sample_id[1]]
                    pose3 = train_dataset.render_poses[poses_sample_id[2]]

                    if args.decouple_interp_range:
                        s12_rot, s3_rot = np.random.uniform(0, args.interp_upbound_rot, size=2)
                        s12_trans, s3_trans = np.random.uniform(0, args.interp_upbound_trans, size=2)
                        s12 = [s12_rot, s12_trans]
                        s3 = [s3_rot, s3_trans]

                    else:
                        if args.sample_based_on_depth:
                            s12, s3 = np.random.beta(args.beta, args.beta, size=2) * args.interp_upbound_rot
                        else:
                            s12, s3 = np.random.uniform(0, args.interp_upbound, size=2)

                    pose = interp3(pose1, pose2, pose3, s12, s3)  # [4, 4]
                    pose = pose.flatten().unsqueeze(0)  # [1, 16]

                    camera_orig = data['camera']  # [1, 34]
                    pose = pose.to(camera_orig)

                    camera_new = torch.cat([camera_orig[:, :18], pose], dim=1)
                    data['camera'] = camera_new


                if args.perturb_camera:
                    rot_trans = transform_src_cameras(src_cameras_orig, rot_param, trans_param,
                                                      args.num_source_views)  # [num_source_views, 3, 4]
                    rot_trans = rot_trans.reshape(-1, 12)  # [num_source_views, 12]
                    src_ray_batch['src_cameras'] = torch.cat(
                        [src_cameras_orig[:, :, :-16], rot_trans.unsqueeze(0), src_cameras_orig[:, :, -4:]], dim=2)

                if args.use_adam:
                    loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data,
                                                           return_loss=True, criterion=criterion)

                    opt.zero_grad()

                    if args.use_pcgrad:
                        opt.pc_backward(loss_dict, major_loss=args.major_loss)
                    else:
                        loss.backward()

                    delta.grad.data *= -1

                    if args.perturb_camera and not args.perturb_camera_no_opt:
                        rot_param.grad.data *= -1
                        trans_param.grad.data *= -1

                    opt.step()
                    scheduler.step()

                else:
                    loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data,
                                                           return_loss=True, criterion=criterion)

                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = delta.data + alpha * torch.sign(grad)
                    delta.grad.zero_()

                    if args.perturb_camera:
                        grad = rot_param.grad.detach()
                        rot_param.data = rot_param.data + args.adv_lr * torch.sign(grad)
                        rot_param.grad.zero_()

                        grad = trans_param.grad.detach()
                        trans_param.data = trans_param.data + args.adv_lr * torch.sign(grad)
                        trans_param.grad.zero_()

                delta.data = clamp(delta.data, -epsilon, epsilon)
                delta.data = clamp(delta.data, lower_limit - src_ray_batch['src_rgbs'],
                                   upper_limit - src_ray_batch['src_rgbs'])

                if args.perturb_camera:
                    rot_param.data = clamp(rot_param.data, torch.tensor(-args.rot_epsilon / 180 * np.pi).cuda(),
                                           torch.tensor(args.rot_epsilon / 180 * np.pi).cuda())
                    trans_param.data = clamp(trans_param.data, torch.tensor(-args.trans_epsilon).cuda(),
                                             torch.tensor(args.trans_epsilon).cuda())

                iters += 1
                if iters > args.adv_iters:
                    continue_flag = False
                    break

    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("Results will be saved at eval/{}.".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    result_dict = {}

    while True:
        try:
            data = next(iterator)
        except:
            break
        if args.local_rank == 0:
            ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = ray_sampler.H, ray_sampler.W
            gt_img = ray_sampler.rgb.reshape(H, W, 3)
            psnr_curr_img, lpips_curr_img, ssim_curr_img = log_view(
                indx,
                args,
                model,
                ray_sampler,
                projector,
                gt_img,
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                ret_alpha=args.ret_alpha,
                single_net=args.single_net,
                delta=delta,
                src_ray_batch_glb=src_ray_batch_glb,
                data=data,
                out_scene_dir=out_scene_dir,
                criterion=criterion
            )
            result_name = "val_{:03d}_coarse.png".format(indx)
            result_dict[result_name] = {}
            result_dict[result_name]['psnr_curr_img'] = psnr_curr_img
            result_dict[result_name]['lpips_curr_img'] = lpips_curr_img
            result_dict[result_name]['ssim_curr_img'] = ssim_curr_img
            psnr_scores.append(psnr_curr_img)
            lpips_scores.append(lpips_curr_img)
            ssim_scores.append(ssim_curr_img)
            torch.cuda.empty_cache()
            indx += 1
    print("Average PSNR: ", np.mean(psnr_scores))
    print("Average LPIPS: ", np.mean(lpips_scores))
    print("Average SSIM: ", np.mean(ssim_scores))
    result_dict['avg_psnr'] = np.mean(psnr_scores)
    result_dict['avg_lpips'] = np.mean(lpips_scores)
    result_dict['avg_ssim'] = np.mean(ssim_scores)

    f = open(os.path.join(out_folder, "results.txt"), "w")
    f.write(str(result_dict))
    f.close()


def log_view(
        global_step,
        args,
        model,
        ray_sampler,
        projector,
        gt_img,
        render_stride=1,
        prefix="",
        out_folder="",
        ret_alpha=False,
        single_net=True,
        delta=0,
        src_ray_batch_glb=None,
        data=None,
        out_scene_dir="",
        criterion=None
):
    model.switch_to_eval()
    ray_batch = ray_sampler.get_all()

    if src_ray_batch_glb is None:  ## if src_ray_batch_glb is not set (i.e., view_specific), then src_ray_batch is the current ray_batch
        src_ray_batch = ray_batch
    else:
        src_ray_batch = src_ray_batch_glb

    if not args.no_attack and args.view_specific and (
            not args.use_trans_attack or global_step == 0):  # optimize view-specific adv peturb
        epsilon = torch.tensor(args.epsilon / 255.).cuda()
        alpha = torch.tensor(args.adv_lr / 255.).cuda()
        upper_limit = 1
        lower_limit = 0

        if args.perturb_camera:
            rot_param = torch.zeros(args.num_source_views, 3)  ## 3 rotation degrees
            rot_param.uniform_(-args.rot_epsilon / 180 * np.pi, args.rot_epsilon / 180 * np.pi)
            rot_param.requires_grad = True

            trans_param = torch.zeros(args.num_source_views, 3)  ## 3 translation distances
            trans_param.uniform_(-args.trans_epsilon, args.trans_epsilon)
            trans_param.requires_grad = True

            src_cameras_orig = src_ray_batch['src_cameras'].clone()  # [1, num_source_views, 34]

        delta = init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit)

        if args.use_adam:
            if args.perturb_camera:
                params = [delta, rot_param, trans_param]
            else:
                params = [delta]

            opt = torch.optim.Adam(params, lr=args.adam_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)

            if args.use_pcgrad:
                opt = PCGrad(opt, num_source_views=args.num_source_views)

        print("Start adversarial perturbation...")
        for num_iter in range(args.adv_iters):
            if args.perturb_camera:
                rot_trans = transform_src_cameras(src_cameras_orig, rot_param, trans_param,
                                                  args.num_source_views)  # [num_source_views, 3, 4]
                rot_trans = rot_trans.reshape(-1, 12)  # [num_source_views, 12]
                src_ray_batch['src_cameras'] = torch.cat(
                    [src_cameras_orig[:, :, :-16], rot_trans.unsqueeze(0), src_cameras_orig[:, :, -4:]], dim=2)

            if args.use_adam:
                loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data,
                                                       return_loss=True, criterion=criterion)

                opt.zero_grad()

                if args.use_pcgrad:
                    opt.pc_backward(loss_dict, major_loss=args.major_loss)
                else:
                    loss.backward()

                delta.grad.data *= -1

                if args.perturb_camera and not args.perturb_camera_no_opt:
                    rot_param.grad.data *= -1
                    trans_param.grad.data *= -1

                opt.step()
                scheduler.step()

            else:
                loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data,
                                                       return_loss=True, criterion=criterion)

                loss.backward()
                grad = delta.grad.detach()
                delta.data = delta.data + alpha * torch.sign(grad)
                delta.grad.zero_()

                if args.perturb_camera:
                    grad = rot_param.grad.detach()
                    rot_param.data = rot_param.data + args.adv_lr * torch.sign(grad)
                    rot_param.grad.zero_()

                    grad = trans_param.grad.detach()
                    trans_param.data = trans_param.data + args.adv_lr * torch.sign(grad)
                    trans_param.grad.zero_()

            delta.data = clamp(delta.data, -epsilon, epsilon)
            delta.data = clamp(delta.data, lower_limit - src_ray_batch['src_rgbs'],
                               upper_limit - src_ray_batch['src_rgbs'])

            if args.perturb_camera:
                rot_param.data = clamp(rot_param.data, torch.tensor(-args.rot_epsilon / 180 * np.pi).cuda(),
                                       torch.tensor(args.rot_epsilon / 180 * np.pi).cuda())
                trans_param.data = clamp(trans_param.data, torch.tensor(-args.trans_epsilon).cuda(),
                                         torch.tensor(args.trans_epsilon).cuda())

    if args.export_adv_source_img:
        adv_src_rgbs = src_ray_batch['src_rgbs'] + delta  # [1, N_views, H, W, 3]
        adv_src_rgbs = adv_src_rgbs[0]  # [N_views, H, W, 3]
        src_rgbs = src_ray_batch['src_rgbs'][0]  # [N_views, H, W, 3]

        for i in range(adv_src_rgbs.shape[0]):
            adv_src_img = adv_src_rgbs[i, :, :, :]
            adv_src_img = (255 * np.clip(adv_src_img.data.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, 'adv_src_{}.png'.format(i)), adv_src_img)

            src_img = src_rgbs[i, :, :, :]
            src_img = (255 * np.clip(src_img.data.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, 'src_{}.png'.format(i)), src_img)

        sys.exit()

    if args.use_purification:
        purif_epsilon = torch.tensor(args.purif_epsilon / 255.).cuda()
        purif_lr = torch.tensor(args.purif_lr / 255.).cuda()
        upper_limit = 1
        lower_limit = 0

        purif = torch.zeros_like(src_ray_batch['src_rgbs'])  # ray_batch['src_rgbs'].shape=[1, N_views, H, W, 3]
        purif.uniform_(-purif_epsilon, purif_epsilon)
        purif.data = clamp(purif, lower_limit - (src_ray_batch['src_rgbs'] + delta),
                           upper_limit - (src_ray_batch['src_rgbs'] + delta))
        purif.requires_grad = True

        opt_purif = torch.optim.Adam([purif], lr=args.adam_lr)
        scheduler_purif = torch.optim.lr_scheduler.StepLR(opt_purif, step_size=args.lr_step_size, gamma=args.lr_gamma)

        print("Start purification...")
        for _ in range(args.purif_iters):
            loss, loss_dict = optimize_purif(args, purif, delta, model, projector, src_ray_batch, data,
                                             self_purification=args.use_self_purification)

            opt_purif.zero_grad()
            loss.backward()
            opt_purif.step()
            scheduler_purif.step()

            purif.data = clamp(purif.data, -purif_epsilon, purif_epsilon)
            purif.data = clamp(purif, lower_limit - (src_ray_batch['src_rgbs'] + delta),
                               upper_limit - (src_ray_batch['src_rgbs'] + delta))

        delta.data = delta.data + purif.data

    if args.def_random_noise > 0:
        delta.data = delta.data + torch.randn_like(delta.data) * args.def_random_noise / 255

    with torch.no_grad():
        if model.feature_net is not None:
            # featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            featmaps = model.feature_net((src_ray_batch['src_rgbs'] + delta).squeeze(0).permute(0, 3, 1, 2))

            if args.use_clean_color or args.use_clean_density:
                featmaps_clean = model.feature_net((src_ray_batch['src_rgbs']).squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps_clean = None

        else:
            featmaps = [None, None]

        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            ret_alpha=args.ret_alpha,
            single_net=single_net,
            args=args,
            featmaps_clean=featmaps_clean,
            src_ray_batch=src_ray_batch,
        )

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_coarse = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_coarse = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        depth_coarse = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        if "depth" in ret["outputs_fine"].keys():
            depth_pred = ret["outputs_fine"]["depth"].detach().cpu()
            depth_fine = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        rgb_fine = None
        depth_fine = None

    rgb_coarse = rgb_coarse.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
    imageio.imwrite(filename, rgb_coarse)

    if depth_coarse is not None:
        depth_coarse = depth_coarse.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(
            out_folder, prefix[:-1] + "_{:03d}_coarse_depth.png".format(global_step)
        )
        imageio.imwrite(filename, depth_coarse)

    if rgb_fine is not None:
        rgb_fine = rgb_fine.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_fine.png".format(global_step))
        imageio.imwrite(filename, rgb_fine)

    if depth_fine is not None:
        depth_fine = depth_fine.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(
            out_folder, prefix[:-1] + "_{:03d}_fine_depth.png".format(global_step)
        )
        imageio.imwrite(filename, depth_fine)

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )
    pred_rgb = torch.clip(pred_rgb, 0.0, 1.0)
    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    return psnr_curr_img, lpips_curr_img, ssim_curr_img


if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()

    args.det = True  ## always use deterministic sampling for coarse and fine samples
    args.run_val = True

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eval(args)
