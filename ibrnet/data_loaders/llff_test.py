# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
import torch
import sys
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class LLFFTestDataset(Dataset):
    def __init__(self, args, mode, scenes=(), use_glb_src=False, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/nerf_llff_data/')
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = args.random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        
        self.train_depth_files = []
        self.render_depth_files = []
        
        self.test_poses = []
        
        self.use_glb_src = use_glb_src

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=args.llff_factor)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = np.arange(poses.shape[0])[::self.args.llffhold]
            i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                (j not in i_test and j not in i_test)])

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test
            
            self.test_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_test]])

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
            
            if args.gt_depth_path:
                depth_dir = os.path.join(args.gt_depth_path, scene)
                assert os.path.exists(depth_dir)
                fnames = sorted(os.listdir(depth_dir))
                depth_files = []
                for fname in fnames:
                    if fname.endswith('.npy'):
                        depth_files.append(os.path.join(depth_dir, fname))
                
                self.train_depth_files.extend(np.array(depth_files)[i_train].tolist())
                self.render_depth_files.extend(np.array(depth_files)[i_render].tolist())


    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        if self.use_glb_src:
            ref_position = np.mean(train_poses[..., 3], axis=0, keepdims=True)
            dist = np.sum(np.abs(train_poses[..., 3] - ref_position), axis=-1)
            nearest_pose_ids = np.argsort(dist)[:num_select]
        
        else:
            nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                    train_poses,
                                                    min(self.num_source_views*subsample_factor, 28),
                                                    tar_id=id_render,
                                                    angular_dist_method='dist')
            nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        src_depths = []
                
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)
            
            if len(self.train_depth_files) > 0:
                src_depth = np.load(self.train_depth_files[id])
                src_depths.append(src_depth)
        
        if len(src_depths) > 0:
            src_depths = np.stack(src_depths, axis=0)   # [num_source_views, H, W]

        if len(self.render_depth_files) > 0:
            depth = np.load(self.render_depth_files[idx])
            
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        if self.mode == 'train' and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            
            # TODO: depth is not processed here
            if len(src_depths) > 0:
                rgb, camera, src_rgbs, src_cameras, src_depths = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                                (crop_h, crop_w), src_depths=src_depths)
            else:
                rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                                (crop_h, crop_w))

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])
        
        data = {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                # 'depth': None if len(self.render_depth_files)==0 else torch.from_numpy(depth),
                # 'src_depths': None if len(self.train_depth_files)==0 else torch.from_numpy(src_depths)
                }
        
        if len(self.render_depth_files) > 0:
            data['depth'] = torch.from_numpy(depth)

        if len(self.train_depth_files) > 0:
            data['src_depths'] = torch.from_numpy(src_depths)
        
        return data
