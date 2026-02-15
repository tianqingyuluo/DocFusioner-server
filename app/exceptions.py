"""
统一异常定义。

所有 Batch 2+ 模块的自定义异常集中定义在此文件。
各异常类使用 @dataclass 定义，继承公共基类 AppError。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppError(Exception):
    """所有应用异常的公共基类。"""

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class LLMError(AppError):
    """LLM 适配层异常基类。"""


@dataclass
class LLMAuthError(LLMError):
    """API Key 无效 (401)。"""

    provider: str = ""
    model: str = ""


@dataclass
class LLMConnectionError(LLMError):
    """网络超时/连接失败。"""

    provider: str = ""
    base_url: str = ""


@dataclass
class LLMModelNotFoundError(LLMError):
    """模型不存在 (404)。"""

    provider: str = ""
    model: str = ""


@dataclass
class LLMOutputParseError(LLMError):
    """JSON 解析/schema 不匹配。"""

    reason: str = ""
    provider: str = ""
    model: str = ""
    raw_output: str = ""


@dataclass
class LLMRateLimitError(LLMError):
    """重试耗尽后仍 429。"""

    provider: str = ""
    retry_count: int = 0


@dataclass
class FileError(AppError):
    """文件工具异常基类。"""


@dataclass
class UnsupportedFileTypeError(FileError):
    """扩展名不在白名单。"""

    file_type: str = ""


@dataclass
class FileTooLargeError(FileError):
    """超过 MAX_UPLOAD_SIZE。"""

    filename: str = ""
    size: int = 0
    max_size: int = 0


@dataclass
class FileDeleteError(FileError):
    """删除文件失败。"""

    path: str = ""
    reason: str = ""


@dataclass
class FileMagicMismatchError(FileError):
    """扩展名与 magic bytes 不匹配。"""

    filename: str = ""
    expected: str = ""
    actual_header: str = ""


@dataclass
class ChromaError(AppError):
    """向量库异常基类。"""


@dataclass
class ChromaUpsertError(ChromaError):
    """批量写入部分/全部失败。"""

    failed_ids: list[str] = field(default_factory=list)
    total: int = 0


@dataclass
class ParseError(AppError):
    """文件完全不可解析。"""

    filename: str = ""
    reason: str = ""
