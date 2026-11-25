"""sounddevice 的空替身。

mediapipe 导入时会顺带导入 sounddevice，后者在没有声卡的服务器上初始化失败。
SoulX-FlashHead 只用 mediapipe 做人脸检测，用不到音频设备。
"""
