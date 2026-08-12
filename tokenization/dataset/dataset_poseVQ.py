import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import tqdm
from smplx import SMPLH, SMPLX
from utils.rotation_conversions import axis_angle_to_matrix
from utils.skeleton import get_smplx_body_parts

def get_dataloader(hparams, split, shuffle=True):

    batch_size = hparams.DATA.BATCH_SIZE
    debug = hparams.EXP.DEBUG
    data_root = hparams.DATA.DATA_ROOT
    mask_body_parts = hparams.DATA.MASK_BODY_PARTS
    rot_type = hparams.ARCH.ROT_TYPE
    debug = hparams.EXP.DEBUG
    smpl_type = hparams.ARCH.SMPL_TYPE
    
    # Dynamically reads from config
    num_workers = hparams.DATA.NUM_WORKERS
    cache_smpl = getattr(hparams.DATA, 'CACHE_SMPL', True)
    pin_memory = getattr(hparams.DATA, 'PIN_MEMORY', True)

    if split == 'train':
        ds_list = hparams.DATA.TRAINLIST.split('_')
        partition = [1] if len(ds_list) == 1 else hparams.DATA.TRAIN_PART.split('_')
        assert len(ds_list) == len(partition), "Number of datasets and parition does not match"
    elif split == 'val':
        ds_list = hparams.DATA.VALLIST.split('_')
        partition = [1/len(ds_list)]*len(ds_list)
    elif split == 'test':
        ds_list = hparams.DATA.TESTLIST.split('_')
        partition = [1/len(ds_list)]*len(ds_list)

    print(f'List of datasets for {split} --> {ds_list} with shuffle = {shuffle} (cache_smpl={cache_smpl})')

    if len(ds_list) == 1:
        dataset = VQPoseDataset(ds_list[0], split, data_root, rot_type, smpl_type, mask_body_parts, debug, cache_smpl=cache_smpl)
    else:
        if split == 'train':
            dataset = MixedTrainDataset(ds_list, partition, split, data_root, rot_type, smpl_type, mask_body_parts, debug, cache_smpl=cache_smpl)
        else:
            dataset = ValDataset(ds_list, split, data_root, rot_type, smpl_type, debug, cache_smpl=cache_smpl)

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        drop_last=True)
    if split == 'train':
        return cycle(loader)
    else:
        return loader

class MixedTrainDataset(data.Dataset):

    def __init__(self, ds_list, partition, split, data_root, rot_type, smpl_type, mask_body_parts, debug, cache_smpl=True):

        self.ds_list = ds_list
        partition = [float(part) for part in partition]
        self.partition = np.array(partition).cumsum()

        self.datasets = [VQPoseDataset(ds, split, data_root, rot_type, smpl_type, mask_body_parts, debug, cache_smpl=cache_smpl) for ds in ds_list]
        self.length = max([len(ds) for ds in self.datasets])

    def __getitem__(self, index):
            p = np.random.rand()
            for i in range(len(self.ds_list)):
                if p <= self.partition[i]:
                    return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length

class VQPoseDataset(data.Dataset):
    def __init__(self, dt, split= 'train', data_root='', rot_type = 'rotmat', smpl_type= 'smplx', mask_body_parts = False, debug = False, cache_smpl=True):

        self.data_root = pjoin(data_root, smpl_type, split)
        self.joints_num = 21
        self.smplx_body_parts = get_smplx_body_parts()
        self.mask_body_parts = mask_body_parts
        self.split = split
        self.smpl_type = smpl_type
        self.rot_type = rot_type
        self.dataset_name = f'_{dt}'
        self.cache_smpl = cache_smpl

        data_file = pjoin(self.data_root, f'{split}_{dt}.npz')
        if not os.path.isfile(data_file):
            raise FileNotFoundError(
                f"Missing tokenization dataset file: {data_file}. "
                f"Expected structure: <DATA_ROOT>/{smpl_type}/{split}/{split}_{dt}.npz. "
                f"Set DATA.DATA_ROOT in config to the folder that contains '{smpl_type}/'."
            )
        data = np.load(data_file)
        total_samples = data['pose_body'].shape[0]

        random_idx = None
        if debug:
            debug_data_length = 8
            random_idx = np.random.choice(total_samples, size=debug_data_length, replace=False)
            print(f'In debug mode, processing with less data')

        raw_pose_body = data['pose_body'][random_idx] if random_idx is not None else data['pose_body']

        if cache_smpl:
            self.smpl_model = eval(f'{smpl_type.upper()}')(f'../data/body_models/{smpl_type}', num_betas=10, ext='pkl')

            print(f"Processing {dt} for {split} with {raw_pose_body.shape[0]} samples. Pre-computing SMPL on GPU...")

            # --- GPU PRE-COMPUTATION BLOCK ---
            self.pose_body_aa = torch.from_numpy(raw_pose_body).float()

            smpl_gpu = self.smpl_model.cuda()

            v_list, j_list, rot_list = [], [], []
            bs = 4096 # Process in massive chunks on the 5090

            with torch.no_grad():
                for i in tqdm.tqdm(range(0, len(self.pose_body_aa), bs), desc=f"Caching {dt}"):
                    batch_aa = self.pose_body_aa[i:i+bs].cuda()
                    curr_bs = batch_aa.shape[0] # Get dynamic batch size (last batch might be < 4096)

                    # Dynamically expand all possible SMPL default parameters to match the current batch size
                    kwargs = {}
                    for param_name in ['global_orient', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                        if hasattr(smpl_gpu, param_name):
                            val = getattr(smpl_gpu, param_name)
                            if val is not None:
                                kwargs[param_name] = val.expand(curr_bs, -1)

                    # Pass the dynamically expanded parameters
                    out = smpl_gpu(body_pose=batch_aa.view(curr_bs, -1), **kwargs)

                    # Move to CPU and cast to float16 to save huge amounts of system RAM
                    v_list.append(out.vertices.cpu().half())
                    j_list.append(out.joints.cpu().half())
                    rot_list.append(axis_angle_to_matrix(batch_aa.view(-1, 21, 3)).cpu().half())

            self.body_vertices = torch.cat(v_list, dim=0)
            self.body_joints = torch.cat(j_list, dim=0)
            self.gt_pose_body = torch.cat(rot_list, dim=0)
            # ---------------------------------
        else:
            # Low-RAM path: keep only the axis-angle poses. SMPL is run per-batch on GPU
            # in the training/eval loops via the shared `body_model` (SMPLHLayer).
            print(f"Processing {dt} for {split} with {raw_pose_body.shape[0]} samples. SMPL cache disabled (CACHE_SMPL=False).")
            self.pose_body_aa = torch.from_numpy(np.ascontiguousarray(raw_pose_body)).float()

    def __len__(self):
        return self.pose_body_aa.shape[0]

    def __getitem__(self, index):
        if self.cache_smpl:
            # __getitem__ is now an instantaneous memory lookup!
            return {
                'pose_body_aa': self.pose_body_aa[index],
                'body_vertices': self.body_vertices[index].float(), # Cast back to float32 for training
                'body_joints': self.body_joints[index].float(),
                'gt_pose_body': self.gt_pose_body[index].float(),
                'dataset_name': self.dataset_name
            }
        return {
            'pose_body_aa': self.pose_body_aa[index],
            'dataset_name': self.dataset_name,
        }


class ValDataset(data.Dataset):
    def __init__(self, dataset_list, split= 'val', data_root='', rot_type = 'rotmat', smpl_type = 'smplx', debug = False, cache_smpl=True):

        self.data_root = pjoin(data_root, smpl_type, split)
        self.joints_num = 21
        self.smplx_body_parts = get_smplx_body_parts()
        self.split = split
        self.smpl_type = smpl_type
        self.rot_type = rot_type
        self.cache_smpl = cache_smpl

        raw_pose_body = np.empty((0,63))
        self.dataset_name = ''

        for dt in dataset_list:
            self.dataset_name += f'_{dt}'
            data_file = pjoin(self.data_root, f'{split}_{dt}.npz')
            if not os.path.isfile(data_file):
                raise FileNotFoundError(
                    f"Missing tokenization dataset file: {data_file}. "
                    f"Expected structure: <DATA_ROOT>/{smpl_type}/{split}/{split}_{dt}.npz. "
                    f"Set DATA.DATA_ROOT in config to the folder that contains '{smpl_type}/'."
                )
            data = np.load(data_file)
            raw_pose_body = np.append(raw_pose_body, data['pose_body'], axis=0)
            print(f"Loaded {dt} for {split} with {data['pose_body'].shape[0]} samples...")

        if debug:
            debug_data_length = 600
            random_idx = np.random.choice(raw_pose_body.shape[0], size=debug_data_length, replace=False)
            print(f'In debug mode, processing with less data')
            raw_pose_body = raw_pose_body[random_idx]

        if cache_smpl:
            self.smpl_model = eval(f'{smpl_type.upper()}')(f'../data/body_models/{smpl_type}', num_betas=10, ext='pkl')

            print(f"Total Val samples: {raw_pose_body.shape[0]}. Pre-computing SMPL on GPU...")

            # --- GPU PRE-COMPUTATION BLOCK ---
            self.pose_body_aa = torch.from_numpy(raw_pose_body).float()

            smpl_gpu = self.smpl_model.cuda()
            v_list, j_list, rot_list = [], [], []
            bs = 4096

            with torch.no_grad():
                for i in tqdm.tqdm(range(0, len(self.pose_body_aa), bs), desc="Caching Val"):
                    batch_aa = self.pose_body_aa[i:i+bs].cuda()
                    curr_bs = batch_aa.shape[0]

                    # Dynamically expand all possible SMPL default parameters to match the current batch size
                    kwargs = {}
                    for param_name in ['global_orient', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                        if hasattr(smpl_gpu, param_name):
                            val = getattr(smpl_gpu, param_name)
                            if val is not None:
                                kwargs[param_name] = val.expand(curr_bs, -1)

                    # Pass the dynamically expanded parameters
                    out = smpl_gpu(body_pose=batch_aa.view(curr_bs, -1), **kwargs)

                    v_list.append(out.vertices.cpu().half())
                    j_list.append(out.joints.cpu().half())
                    rot_list.append(axis_angle_to_matrix(batch_aa.view(-1, 21, 3)).cpu().half())

            self.body_vertices = torch.cat(v_list, dim=0)
            self.body_joints = torch.cat(j_list, dim=0)
            self.gt_pose_body = torch.cat(rot_list, dim=0)
            # ---------------------------------
        else:
            print(f"Total Val samples: {raw_pose_body.shape[0]}. SMPL cache disabled (CACHE_SMPL=False).")
            self.pose_body_aa = torch.from_numpy(np.ascontiguousarray(raw_pose_body)).float()

    def __len__(self):
        return self.pose_body_aa.shape[0]

    def __getitem__(self, index):
        if self.cache_smpl:
            return {
                'pose_body_aa': self.pose_body_aa[index],
                'body_vertices': self.body_vertices[index].float(),
                'body_joints': self.body_joints[index].float(),
                'gt_pose_body': self.gt_pose_body[index].float(),
                'dataset_name': self.dataset_name
            }
        return {
            'pose_body_aa': self.pose_body_aa[index],
            'dataset_name': self.dataset_name,
        }

def cycle(iterable):
    while True:
        for x in iterable:
            yield x