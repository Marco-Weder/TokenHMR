import torch


def get_smplx_body_parts():
    return {
        0: [11, 14],                    # head
        1: [12, 15, 17, 19],            # left-arm
        2: [13, 16, 18, 20],            # right-arm
        3: [0, 3, 6, 9],                # left-leg
        4: [1, 4, 7, 10],               # right-leg
    }


# SMPL-H kinematic tree for the 21 body joints (root/pelvis excluded). Entry i is the
# parent index of joint i; -1 means the parent is the root pelvis. This is the same tree
# the transformer's USE_KINEMATIC_PE relies on, kept here as the single source of truth so
# both the kinematic positional encoding and the GNN tokenizer build edges from one list.
SMPLH_PARENTS_21 = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11, 12, 13, 15, 16, 17, 18]


def build_skeleton_attention_mask(num_joints: int = 21, n_hops: int = 1,
                                  connect_pelvis_siblings: bool = True) -> torch.Tensor:
    """Boolean (num_joints, num_joints) attention mask for the skeleton-masked transformer.

    ``mask[i, j] = True`` means joint i may attend to joint j. Restricting self-attention to
    this mask turns a transformer layer into a GAT-style graph layer on the kinematic tree.

    - Edges are the undirected parent links from `SMPLH_PARENTS_21`.
    - `connect_pelvis_siblings`: the pelvis root is excluded from the 21 joints, so without it
      the graph is 3 disconnected components (left leg, right leg, upper body). The three
      pelvis children (L_Hip=0, R_Hip=1, Spine1=2) are all anatomically adjacent to the pelvis,
      so they are connected pairwise to restore a single connected body graph.
    - Self-loops are always included, so no softmax row is fully masked (no NaNs).
    - `n_hops`: closure to the n-hop neighbourhood (n_hops=2 -> attend within 2 bones).
    """
    parents = SMPLH_PARENTS_21[:num_joints]
    A = torch.zeros(num_joints, num_joints, dtype=torch.bool)
    for i, p in enumerate(parents):
        if p >= 0:
            A[i, p] = True
            A[p, i] = True
    if connect_pelvis_siblings:
        roots = [i for i, p in enumerate(parents) if p < 0]
        for i in roots:
            for j in roots:
                if i != j:
                    A[i, j] = True
    A = A | torch.eye(num_joints, dtype=torch.bool)
    mask = A
    for _ in range(max(1, int(n_hops)) - 1):
        mask = (mask.float() @ A.float()) > 0
    return mask


def build_skeleton_adjacency(num_joints: int = 21, add_self_loops: bool = True,
                             normalize: bool = True) -> torch.Tensor:
    """Build the (num_joints, num_joints) skeleton adjacency for the GNN tokenizer.

    Each joint is connected (undirected) to its parent in `SMPLH_PARENTS_21`. With the
    defaults this returns the symmetric-normalized adjacency with self-loops,
    ``A_norm = D^{-1/2} (A + I) D^{-1/2}`` — the standard GCN propagation matrix
    (Kipf & Welling, 2017). A graph-conv layer aggregates each joint's neighbours via
    ``A_norm @ H``, so messages only flow along anatomical connections.
    """
    parents = SMPLH_PARENTS_21[:num_joints]
    A = torch.zeros(num_joints, num_joints)
    for i, p in enumerate(parents):
        if p >= 0:
            A[i, p] = 1.0
            A[p, i] = 1.0
    if add_self_loops:
        A = A + torch.eye(num_joints)
    if normalize:
        deg = A.sum(dim=1).clamp(min=1.0)
        d_inv_sqrt = deg.pow(-0.5)
        A = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
    return A
