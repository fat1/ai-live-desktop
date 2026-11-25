"""持续向本机 llama-server 发送对话请求，统计生成速度。用于多模型并发测试。

用法::

    python bench_llm_loop.py --url http://127.0.0.1:8089 --seconds 120

只依赖标准库。
"""

import argparse
import json
import time
import urllib.request

QUESTIONS = [
    "这个面霜敏感肌能用吗？",
    "多久能发货？包邮吗？",
    "主播身上这件衣服有链接吗？",
    "和上一款比有什么区别？",
    "现在下单有什么优惠？",
]
SYSTEM_PROMPT = "你是一名带货主播，用口语化的中文简短回答观众弹幕，每次不超过两句话。"


def ask(url: str, question: str) -> dict:
    """发送一次对话请求。

    参数:
        url: llama-server 地址
        question: 观众弹幕
    返回:
        服务端返回的 JSON（含 choices 和 timings）
    抛出:
        urllib.error.URLError: 服务不可达时
    """
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        "max_tokens": 120,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8089")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.0, help="两次请求之间的间隔秒数，0 表示满负荷")
    parser.add_argument("--report_every", type=int, default=5, help="每多少次回复输出一次累计结果")
    args = parser.parse_args()

    deadline = time.perf_counter() + args.seconds
    count = tokens = 0
    generation_ms = 0.0
    latencies = []
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        reply = ask(args.url, QUESTIONS[count % len(QUESTIONS)])
        latencies.append(time.perf_counter() - start)
        timings = reply.get("timings", {})
        tokens += timings.get("predicted_n", 0)
        generation_ms += timings.get("predicted_ms", 0.0)
        if count == 0:
            print("sample reply:", reply["choices"][0]["message"]["content"].strip(), flush=True)
        count += 1
        time.sleep(args.interval)
        if count % args.report_every == 0:  # 定期输出累计结果，进程被中途结束时也能拿到数据
            speed = tokens / (generation_ms / 1000) if generation_ms else 0.0
            print(f"llm: {count} replies, {speed:.1f} tok/s, mean latency "
                  f"{sum(latencies) / len(latencies):.2f}s, max latency {max(latencies):.2f}s", flush=True)


if __name__ == "__main__":
    main()
