"""单显卡下的固定返回值，见上级包的说明。"""


def get_sequence_parallel_rank() -> int:
    """当前进程在并行组里的序号；单显卡恒为 0。"""
    return 0


def get_sequence_parallel_world_size() -> int:
    """并行组的大小；单显卡恒为 1。"""
    return 1


def get_sp_group() -> None:
    """并行进程组；单显卡没有。"""
    return None
