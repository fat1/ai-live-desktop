"""文本处理：把大模型的流式输出切成句子，并清洗掉不该念出来的内容。只依赖标准库。"""

import re
from collections.abc import Iterable, Iterator

DEFAULT_PERSONA = (
    "你是一名带货主播，正在直播间里和观众聊天。"
    "请用口语化的中文简短回答观众的弹幕，语气亲切自然，每次回答两到三句话，"
    "不要使用表情符号、括号说明或任何格式标记。"
)
SENTENCE_END = re.compile(r"[。！？!?；;\n]")
CLAUSE_END = re.compile(r"[。！？!?；;\n，,、：:]")
MAX_SENTENCE_CHARS = 40
BRACKETED = re.compile(r"[（(\[【<][^）)\]】>]{0,30}[）)\]】>]")
# 只保留中文、字母、数字和常用标点，其余（表情、星号、井号等）一律去掉
UNSPEAKABLE = re.compile(r"[^一-鿿A-Za-z0-9\s，。！？、；：,.!?;:%％·\-—…\"“”'‘’]")
SPEAKABLE = re.compile(r"[一-鿿A-Za-z0-9]")


def stream_sentences(deltas: Iterable[str], min_chars: int = 6, first_clause: bool = True) -> Iterator[str]:
    """把流式输出的文本片段实时切成句子。

    参数:
        deltas: 大模型逐步产出的文本片段
        min_chars: 短于该长度的句子先不产出，和后面的内容合并
        first_clause: 第一段遇到逗号就切出来，让数字人更早开口；之后仍按整句切
    返回:
        逐句产出，保留句末标点
    """
    buffer = ""
    pattern = CLAUSE_END if first_clause else SENTENCE_END
    for delta in deltas:
        buffer += delta
        search_from = 0
        while len(buffer) > MAX_SENTENCE_CHARS and pattern.search(buffer) is None:
            # 一直不出现标点时按字数强制切开，避免缓冲无限增长
            yield buffer[:MAX_SENTENCE_CHARS]
            buffer = buffer[MAX_SENTENCE_CHARS:]
            pattern = SENTENCE_END
        while (match := pattern.search(buffer, search_from)) is not None:
            if match.end() >= min_chars:
                sentence = buffer[:match.end()].strip()
                buffer = buffer[match.end():]
                search_from = 0
                if sentence:
                    pattern = SENTENCE_END  # 只有第一段按逗号切，之后恢复按句切
                    yield sentence
            else:
                search_from = match.end()
    if buffer.strip():
        yield buffer.strip()


def clean_for_speech(text: str) -> str:
    """去掉不该念出来的内容：括号里的说明、表情和各种标记符号。

    小模型不一定遵守提示词里"不要用表情和括号"的要求，送去合成前再清洗一遍。

    返回:
        清洗后的文本；如果没有剩下可念的字（例如只有标点或表情），返回空字符串
    """
    text = BRACKETED.sub("", text)
    text = UNSPEAKABLE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if SPEAKABLE.search(text) else ""
