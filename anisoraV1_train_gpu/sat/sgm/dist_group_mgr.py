import torch.distributed
from datetime import timedelta

class DistGp():
    def __init__(self, name: str):
        self.__group = None
        # Number of members in the group.
        self.__size = None
        self.__name = name
        return
    
    def initialize(self, group_size: int, timeout = timedelta(seconds=1800)):
        assert group_size > 0, f"group_size {group_size} is invalid."
        assert self.__group is None, "group is already initialized."
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        for i in range(0, world_size, group_size):
            ranks = range(i, i + group_size)
            group = torch.distributed.new_group(ranks, timeout=timeout)
            if rank in ranks:
                self.__group = group
                self.__size = group_size
                break
        return
    
    def rank_in_group(self):
        rank = torch.distributed.get_rank()
        return self.show_group().index(rank)

    def is_initialized(self):
        return self.__group != None and self.__size != None and self.__name != None
    
    def show_group(self) -> list:
        rtv = torch.distributed.get_process_group_ranks(self.group)
        return rtv

    @property
    def group(self):
        assert self.__group is not None, f"{self.__name} group is not initialized."
        return self.__group
    
    @group.setter
    def group(self, group_list: list[int]):
        assert group_list is not None, f"Arg is invalid."
        assert torch.distributed.get_rank() in group_list, f"Rank is not in group_list."
        self.__size = len(group_list)
        self.__group = torch.distributed.new_group(group_list)
        return

    @property
    def gp_size(self):
        return self.__size
    
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name: str):
        self.__name = name
        return


class DistGpMgr():
    __group_dict = {}

    @staticmethod
    def add(group_name: str, group_size: int, timeout = timedelta(seconds=1800)):
        val = DistGp(group_name)
        val.initialize(group_size, timeout)
        DistGpMgr.__group_dict[group_name] = val
        return

    @staticmethod
    def get(group_name: str) -> DistGp:
        if group_name not in DistGpMgr.__group_dict:
            return None
        else:
            return DistGpMgr.__group_dict[group_name]
