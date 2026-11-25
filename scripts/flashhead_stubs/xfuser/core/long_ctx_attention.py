"""多显卡注意力的占位：单显卡推理不会走到这里。"""


class xFuserLongContextAttention:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("单显卡替身不支持多显卡注意力")
