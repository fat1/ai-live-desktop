"""实时播放：按固定帧率的时钟出画面和声音。

播放端每 40 毫秒必须给出一帧画面和对应的一小段声音：
- 说话队列里有到点的帧，就播它和它的声音；
- 没有，就播不说话的待机画面和静音。

生成端（大模型、语音、口型）只管往说话队列里放帧，跟不上时观众看到的是待机画面，
而不是画面卡死。每一帧说话画面和它那 40 毫秒声音是绑在一起入队的，所以音画不会错位。
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from engines.lipsync import Avatar, LipSyncEngine


class Sink(Protocol):
    """画面和声音的去向，例如写文件或显示在窗口里。"""

    def write(self, frame: np.ndarray, audio: np.ndarray) -> None:
        """输出一帧 BGR 画面和它对应的一段 int16 声音。"""


@dataclass
class SpeechFrame:
    """一帧说话画面。"""

    index: int            # 对应素材的第几帧，决定这帧应该接在待机画面的哪个位置
    face: np.ndarray      # 口型模型生成的人脸
    audio: np.ndarray     # 这一帧时长内的 int16 声音
    reply: int            # 属于第几条回复


@dataclass
class ReplyStats:
    """一条回复的播放统计。"""

    first_played: float | None = None   # 第一帧说话画面播出的时刻
    frames: int = 0                      # 已播出的说话帧数
    stall_frames: int = 0                # 说到一半时因为下一帧没生成好而插入的待机帧数
    jumps: list[int] = field(default_factory=list)  # 说话帧来晚导致画面回跳的帧数
    done: bool = False                   # 这条回复的帧是否已全部入队


class Player:
    """时钟驱动的播放线程。"""

    def __init__(self, avatar: Avatar, sink: Sink, sample_rate: int, fps: int = 25, min_buffer: int = 8,
                 pause_frames: int = 8) -> None:
        """
        参数:
            avatar: 人物素材
            sink: 画面和声音的去向
            sample_rate: 输出声音的采样率
            fps: 帧率
            min_buffer: 一条回复开始播之前至少要攒够的帧数，用来吸收生成速度的抖动
            pause_frames: 句与句之间正常的停顿帧数；一条回复说到一半时，
                空档超过这个长度才算作卡顿
        """
        self._pause_frames = pause_frames
        self._idle_run = 0        # 当前回复里连续播了多少帧待机画面
        self._avatar = avatar
        self._sink = sink
        self._fps = fps
        self._samples_per_frame = sample_rate // fps
        self._min_buffer = min_buffer
        self._queue: deque[SpeechFrame] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._cursor = 0          # 下一帧要播的素材帧序号
        self._playing: int | None = None  # 正在播的回复编号
        self.stats: dict[int, ReplyStats] = {}
        self.error: BaseException | None = None
        self.frames_played = 0
        self.late_frames = 0      # 出画面晚于时钟一帧以上的次数
        self.max_lag = 0.0        # 出画面落后时钟的最大秒数

    @property
    def cursor(self) -> int:
        """下一帧要播的素材帧序号。生成端据此决定新一句话从素材的哪一帧接上。"""
        return self._cursor

    def backlog(self) -> int:
        """已生成但还没播的说话帧数。"""
        return len(self._queue)

    def put(self, frame: SpeechFrame) -> None:
        """放入一帧说话画面。必须按播放顺序放入。"""
        with self._lock:
            self.stats.setdefault(frame.reply, ReplyStats())
            self._queue.append(frame)

    def finish_reply(self, reply: int) -> None:
        """声明某条回复的帧已全部放入。"""
        with self._lock:
            self.stats.setdefault(reply, ReplyStats()).done = True

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _next(self) -> SpeechFrame | None:
        """决定这一拍播说话帧还是待机帧。返回 None 表示待机。"""
        with self._lock:
            if not self._queue:
                if self._playing is not None:
                    stats = self.stats[self._playing]
                    self._idle_run += 1
                    if stats.done:
                        self._playing = None  # 这条回复播完了
                    elif self._idle_run > self._pause_frames:
                        stats.stall_frames += 1  # 说到一半断了粮，空档超出了正常的句间停顿
                return None
            head = self._queue[0]
            stats = self.stats[head.reply]
            if self._playing != head.reply:
                # 新回复开始前先攒一点缓冲，除非它已经全部生成完
                if len(self._queue) < self._min_buffer and not stats.done:
                    return None
            if head.index > self._cursor and head.index - self._cursor < 50:
                return None  # 还没到这帧该出现的位置，继续播待机画面走过去
            self._queue.popleft()
            if head.index < self._cursor:
                stats.jumps.append(self._cursor - head.index)  # 来晚了，画面往回跳
            if stats.first_played is None:
                stats.first_played = time.perf_counter()
            stats.frames += 1
            self._idle_run = 0
            self._playing = head.reply
            return head

    def _run(self) -> None:
        silence = np.zeros(self._samples_per_frame, dtype=np.int16)
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                speech = self._next()
                if speech is None:
                    index, face = LipSyncEngine.idle(self._avatar, self._cursor)
                    audio = silence
                else:
                    index, face, audio = speech.index, speech.face, speech.audio
                self._sink.write(LipSyncEngine.blend(self._avatar, index, face), audio)
                self._cursor = index + 1
                self.frames_played += 1
                next_tick += 1 / self._fps
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self.max_lag = max(self.max_lag, -delay)  # 这一帧出晚了
                    if -delay > 1 / self._fps:
                        self.late_frames += 1
        except BaseException as error:  # 交给主线程报告
            self.error = error
