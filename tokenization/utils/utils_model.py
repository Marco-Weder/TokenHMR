import logging
import math
import os
import sys

import torch


@torch.no_grad()
def codebook_usage_stats(counts, nb_code):
    """Codebook-usage metrics from a code-count histogram.

    counts:  1-D tensor of length nb_code (how often each code was selected).
    nb_code: codebook size.

    Returns python floats:
      entropy        H(p) = -sum p*log p   (nats). High = uniform usage; 0 = one-hot collapse.
      norm_entropy   H(p) / log(nb_code) in [0, 1]. 1 = perfectly uniform; 0 = one-hot.
      kl_to_uniform  KL(p || uniform) = log(nb_code) - H(p). 0 = uniform; large = collapse.
                     This is the "probability distance" of the usage from uniform.
      codebook_util  percentage of codes used at least once.
    """
    counts = counts.detach().float().reshape(-1)
    total = counts.sum().clamp(min=1.0)
    prob = counts / total
    entropy = -(prob * (prob + 1e-10).log()).sum()
    log_n = math.log(nb_code) if nb_code > 1 else 1.0
    return {
        'entropy': entropy.item(),
        'norm_entropy': (entropy / log_n).item(),
        'kl_to_uniform': (log_n - entropy).item(),
        'codebook_util': ((counts > 0).float().sum() / nb_code * 100.0).item(),
    }


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger



    