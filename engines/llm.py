"""本地大模型：管理 llama-server 进程并提供对话接口。只依赖标准库。"""

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlmServer:
    """启动并管理本机的 llama-server 进程，可作为上下文管理器使用。"""

    def __init__(self, llama_dir: str | Path, model_path: str | Path, port: int = 8089,
                 context_size: int = 2048, gpu_layers: int = 99) -> None:
        """
        参数:
            llama_dir: llama.cpp 可执行文件所在目录
            model_path: GGUF 模型文件
            port: 监听端口，只绑定本机
            context_size: 上下文长度，越小越省显存；直播回复很短，2048 足够
            gpu_layers: 放到显卡上的层数，0 表示纯 CPU 运行
        """
        self.url = f"http://127.0.0.1:{port}"
        self._command = [
            str(Path(llama_dir) / "llama-server.exe"), "-m", str(model_path), "--host", "127.0.0.1",
            "--port", str(port), "-c", str(context_size), "-ngl", str(gpu_layers), "--jinja",
        ]
        self._process: subprocess.Popen | None = None

    def start(self, timeout: float = 120.0) -> None:
        """启动服务并等待模型加载完成。

        抛出:
            RuntimeError: 进程提前退出或超时未就绪
        """
        self._process = subprocess.Popen(self._command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server 启动失败，退出码 {self._process.returncode}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server 在限定时间内没有就绪")

    def stop(self) -> None:
        """结束服务进程。"""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def __enter__(self) -> "LlmServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def chat(self, system_prompt: str, user_message: str, max_tokens: int = 160,
             temperature: float = 0.7) -> str:
        """生成一次回复。

        参数:
            system_prompt: 人设和回答要求
            user_message: 观众弹幕或脚本内容
            max_tokens: 回复的最大长度
            temperature: 随机程度
        返回:
            回复文本
        抛出:
            urllib.error.URLError: 服务不可达
        """
        body = json.dumps({
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_message}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Qwen3 默认会先输出思考过程，直播回复不需要
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            reply = json.loads(response.read().decode("utf-8"))
        return reply["choices"][0]["message"]["content"].strip()
