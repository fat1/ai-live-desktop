"""xfuser 的单显卡替身。

SoulX-FlashHead 在导入时就需要 xfuser（多显卡并行库），而 xfuser 只支持 Linux
（依赖 fcntl）。单显卡推理实际只用到下面几个函数的固定返回值，所以在 Windows 上
用这个包顶替：把 scripts/flashhead_stubs 加到 PYTHONPATH 即可。不支持多显卡。
"""
