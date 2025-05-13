import torch
import torch.distributed as dist
# from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup

from videosys.utils.logging import init_dist_logger, logger


class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, cp_size, sp_size):
        super().__init__(dp_size, cp_size, sp_size)
        dp_axis, cp_axis, sp_axis = 0, 1, 2

        self.dp_size = dp_size
        self.dp_group: ProcessGroup = self.get_group_along_axis(dp_axis)
        self.dp_rank = dist.get_rank(self.dp_group)

        self.cp_size = cp_size
        if cp_size > 1:
            self.cp_group: ProcessGroup = self.get_group_along_axis(cp_axis)
            self.cp_rank = dist.get_rank(self.cp_group)
        else:
            self.cp_group = None
            self.cp_rank = None

        self.sp_size = sp_size
        if sp_size > 1:
            self.sp_group: ProcessGroup = self.get_group_along_axis(sp_axis)
            self.sp_rank = dist.get_rank(self.sp_group)
        else:
            self.sp_group = None
            self.sp_rank = None

        logger.info(f"Init parallel manager with dp_size: {dp_size}, cp_size: {cp_size}, sp_size: {sp_size}")


def initialize(
    rank=0,
    world_size=1,
    init_method=None,
):
    if not dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        init_dist_logger()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
