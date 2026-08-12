import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from external.tokenhmr.tokenhmr.lib.utils.rotation_utils import axis_angle_to_matrix, matrix_to_rotation_6d
from external.tokenhmr.tokenization.models.vanilla_pose_vqvae import EncodeTokens

def calculate_exact_utilization(pkl_path, ckpt_path, codebook_size=2048, batch_size=1024):
    device = torch.device('cuda')
    
    print(f"Loading results from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        results = pkl.load(f)
        
    # Extract the ground-truth axis-angle poses (N, 21, 3)
    gt_aa = torch.tensor(np.array(results['gt_aa']), dtype=torch.float32).to(device)
    total_poses = gt_aa.shape[0]
    print(f"Loaded {total_poses} poses.")

    print("Loading Encoder...")
    encoder = EncodeTokens(ckpt_path=ckpt_path).to(device).eval()

    active_tokens = set()

    print("Encoding poses to track exact token usage...")
    with torch.no_grad():
        for i in tqdm(range(0, total_poses, batch_size)):
            batch_aa = gt_aa[i:i+batch_size] # Shape: (B, 63)
            
            # FIX: Reshape (B, 63) -> (B, 21, 3) so each joint has its own XYZ axis-angle
            batch_aa = batch_aa.view(-1, 21, 3)
            
            # Now the utility can see the '3' it expects
            rotmats = axis_angle_to_matrix(batch_aa) # Shape: (B, 21, 3, 3)
            poses_6d = matrix_to_rotation_6d(rotmats) # Shape: (B, 21, 6)
            
            tokens = encoder(poses_6d)
            
            batch_unique = torch.unique(tokens).cpu().numpy()
            active_tokens.update(batch_unique)

    num_active = len(active_tokens)
    utilization_pct = (num_active / codebook_size) * 100
    
    print("\n" + "="*40)
    print("      CODEBOOK UTILIZATION REPORT      ")
    print("="*40)
    print(f"Total Poses Evaluated : {total_poses}")
    print(f"Codebook Size         : {codebook_size}")
    print(f"Exact Active Tokens   : {num_active}")
    print(f"Utilization %         : {utilization_pct:.2f}%")
    print("="*40)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Exact codebook utilization of a pose-VQ tokenizer, over the '
                    'ground-truth poses of a validation results pickle.',
        epilog='The results pickle is written by utils/eval_poseVQ.py during tokenizer '
               'validation, as results_<VALLIST>.pkl. Regenerate it by running a '
               'validation pass of train_poseVQ.py if it is not on disk.')
    parser.add_argument('--pkl', required=True,
                        help='validation results pickle holding gt_aa, e.g. '
                             'results_HumanEva_HDM05_SFU_MPI-Mosh_MOYO.pkl')
    parser.add_argument('--ckpt', required=True,
                        help='tokenizer checkpoint, e.g. output/<exp>/best_net.pth')
    parser.add_argument('--codebook-size', type=int, default=2048,
                        help='codes in the codebook (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    calculate_exact_utilization(args.pkl, args.ckpt,
                                codebook_size=args.codebook_size,
                                batch_size=args.batch_size)