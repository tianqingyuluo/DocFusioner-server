"""
解析器基类与数据结构。

定义文档解析器的抽象接口、富结构化输出数据结构、注册/工厂机制。
所有数据结构使用 @dataclass 定义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.exceptions import UnsupportedFileTypeError

BlockType = Literal["paragraph", "heading", "table", "list_item", "code", "image_text"]


@dataclass
class ContentBlock:
    """文档内容块。"""

    index: int
    text: str
    block_type: BlockType
    level: int | None = None
    section: str | None = None
    page: int | None = None
    sheet_name: str | None = None
    row_index: int | None = None
    extra: dict | None = None


@dataclass
class DocumentMetadata:
    """文档级解析元数据。"""

    total_pages: int | None = None
    total_sheets: int | None = None
    heading_tree: list[str] | None = None
    encoding: str | None = None


@dataclass
class ParsedDocument:
    """解析器输出：富结构化文档。"""

    filename: str
    file_type: str
    blocks: list[ContentBlock]
    metadata: DocumentMetadata
    warnings: list[str] = field(default_factory=list)


class BaseParser(ABC):
    """解析器抽象基类。"""

    supported_types: list[str]

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        同步解析文件为富结构化文档。

        - 尽量解析更多内容，部分异常记入 warnings
        - 只有完全不可用时才抛 ParseError
        """


_registry: dict[str, type[BaseParser]] = {}


def register_parser(parser_class: type[BaseParser]) -> None:
    """注册解析器类。"""
    for ft in parser_class.supported_types:
        _registry[ft] = parser_class


def get_parser(file_type: str) -> BaseParser:
    """根据文件类型获取解析器实例。"""
    if file_type not in _registry:
        raise UnsupportedFileTypeError(
            message=f"不支持的文件类型: {file_type}",
            file_type=file_type,
        )
    return _registry[file_type]()
