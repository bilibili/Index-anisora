from sgm.dist_group_mgr import DistGpMgr
import torch
# Group label for sequence parallel in DiffusionTransformer.
LABEL_D_TRANSFORMER = "D-transformer"

def get_sequence_parallel_group():
    t_sp = DistGpMgr.get(LABEL_D_TRANSFORMER)
    if t_sp == None:
        return []
    else:
        return t_sp.group

def get_sequence_parallel_size():
    t_sp = DistGpMgr.get(LABEL_D_TRANSFORMER)
    if t_sp == None:
        return 0
    else:
        return t_sp.gp_size

def enable_sequence_parallel():
    return get_sequence_parallel_size() > 1

def get_sequence_parallel_group_rank():
    rank = torch.distributed.get_rank()
    sp_group_rank = rank // get_sequence_parallel_size()
    return sp_group_rank

def get_sequence_parallel_rank():
    rank = torch.distributed.get_rank()
    sp_rank = rank % get_sequence_parallel_size()
    return sp_rank

