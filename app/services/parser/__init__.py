"""
解析器包。

各解析器模块导入时自动注册。
第三批实现具体解析器后，在此显式导入以触发注册。
"""

from app.services.parser.base import (
    BaseParser,
    BlockType,
    ContentBlock,
    DocumentMetadata,
    ParsedDocument,
    get_parser,
    register_parser,
)

__all__ = [
    "BaseParser",
    "BlockType",
    "ContentBlock",
    "DocumentMetadata",
    "ParsedDocument",
    "get_parser",
    "register_parser",
]
