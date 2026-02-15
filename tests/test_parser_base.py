"""测试解析器基类：数据结构 + ABC + 注册/工厂机制。"""

from pathlib import Path

import pytest

from app.exceptions import ParseError, UnsupportedFileTypeError


class TestBlockType:
    """BlockType 类型别名测试"""

    def test_valid_block_types(self):
        from app.services.parser.base import BlockType

        assert BlockType is not None


class TestContentBlock:
    """ContentBlock dataclass 测试"""

    def test_required_fields(self):
        from app.services.parser.base import ContentBlock

        block = ContentBlock(index=0, text="段落内容", block_type="paragraph")
        assert block.index == 0
        assert block.text == "段落内容"
        assert block.block_type == "paragraph"

    def test_optional_fields_default_none(self):
        from app.services.parser.base import ContentBlock

        block = ContentBlock(index=0, text="", block_type="paragraph")
        assert block.level is None
        assert block.section is None
        assert block.page is None
        assert block.sheet_name is None
        assert block.row_index is None
        assert block.extra is None

    def test_heading_with_level(self):
        from app.services.parser.base import ContentBlock

        block = ContentBlock(
            index=1, text="第一章", block_type="heading", level=1, section=None
        )
        assert block.level == 1

    def test_table_with_sheet_name(self):
        from app.services.parser.base import ContentBlock

        block = ContentBlock(
            index=2,
            text="姓名: 张三 | 年龄: 30",
            block_type="table",
            sheet_name="Sheet1",
            row_index=5,
        )
        assert block.sheet_name == "Sheet1"
        assert block.row_index == 5


class TestDocumentMetadata:
    """DocumentMetadata dataclass 测试"""

    def test_all_defaults_none(self):
        from app.services.parser.base import DocumentMetadata

        meta = DocumentMetadata()
        assert meta.total_pages is None
        assert meta.total_sheets is None
        assert meta.heading_tree is None
        assert meta.encoding is None

    def test_pdf_metadata(self):
        from app.services.parser.base import DocumentMetadata

        meta = DocumentMetadata(total_pages=10)
        assert meta.total_pages == 10

    def test_excel_metadata(self):
        from app.services.parser.base import DocumentMetadata

        meta = DocumentMetadata(total_sheets=3)
        assert meta.total_sheets == 3


class TestParsedDocument:
    """ParsedDocument dataclass 测试"""

    def test_required_fields(self):
        from app.services.parser.base import (
            ContentBlock,
            DocumentMetadata,
            ParsedDocument,
        )

        doc = ParsedDocument(
            filename="test.docx",
            file_type="docx",
            blocks=[ContentBlock(index=0, text="hello", block_type="paragraph")],
            metadata=DocumentMetadata(),
        )
        assert doc.filename == "test.docx"
        assert len(doc.blocks) == 1

    def test_warnings_default_empty(self):
        from app.services.parser.base import DocumentMetadata, ParsedDocument

        doc = ParsedDocument(
            filename="test.md",
            file_type="md",
            blocks=[],
            metadata=DocumentMetadata(),
        )
        assert doc.warnings == []

    def test_warnings_accumulate(self):
        from app.services.parser.base import DocumentMetadata, ParsedDocument

        doc = ParsedDocument(
            filename="test.pdf",
            file_type="pdf",
            blocks=[],
            metadata=DocumentMetadata(),
            warnings=["表格提取失败", "第3页为空"],
        )
        assert len(doc.warnings) == 2


class TestBaseParserABC:
    """BaseParser 抽象基类测试"""

    def test_cannot_instantiate_directly(self):
        from app.services.parser.base import BaseParser

        with pytest.raises(TypeError):
            BaseParser()  # type: ignore[abstract]

    def test_concrete_parser_must_implement_parse(self):
        from app.services.parser.base import BaseParser

        class IncompleteParser(BaseParser):
            supported_types = ["txt"]

        with pytest.raises(TypeError):
            IncompleteParser()  # type: ignore[abstract]

    def test_concrete_parser_works(self):
        from app.services.parser.base import (
            BaseParser,
            DocumentMetadata,
            ParsedDocument,
        )

        class DummyParser(BaseParser):
            supported_types = ["txt"]

            def parse(self, file_path: Path) -> ParsedDocument:
                return ParsedDocument(
                    filename=file_path.name,
                    file_type="txt",
                    blocks=[],
                    metadata=DocumentMetadata(),
                )

        parser = DummyParser()
        result = parser.parse(Path("/tmp/test.txt"))
        assert result.filename == "test.txt"


class TestParserRegistry:
    """解析器注册/工厂机制测试"""

    def test_register_and_get_parser(self):
        from app.services.parser.base import (
            BaseParser,
            DocumentMetadata,
            ParsedDocument,
            _registry,
            get_parser,
            register_parser,
        )

        _registry.clear()

        class FakeParser(BaseParser):
            supported_types = ["txt", "md"]

            def parse(self, file_path: Path) -> ParsedDocument:
                return ParsedDocument(
                    filename="", file_type="txt", blocks=[], metadata=DocumentMetadata()
                )

        register_parser(FakeParser)

        parser = get_parser("txt")
        assert isinstance(parser, FakeParser)
        parser2 = get_parser("md")
        assert isinstance(parser2, FakeParser)

    def test_get_parser_unknown_type_raises(self):
        from app.services.parser.base import _registry, get_parser

        _registry.clear()

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            get_parser("exe")
        assert exc_info.value.file_type == "exe"

    def test_register_overwrites_existing(self):
        from app.services.parser.base import (
            BaseParser,
            DocumentMetadata,
            ParsedDocument,
            _registry,
            get_parser,
            register_parser,
        )

        _registry.clear()

        class ParserA(BaseParser):
            supported_types = ["txt"]

            def parse(self, file_path: Path) -> ParsedDocument:
                return ParsedDocument(
                    filename="", file_type="txt", blocks=[], metadata=DocumentMetadata()
                )

        class ParserB(BaseParser):
            supported_types = ["txt"]

            def parse(self, file_path: Path) -> ParsedDocument:
                return ParsedDocument(
                    filename="", file_type="txt", blocks=[], metadata=DocumentMetadata()
                )

        register_parser(ParserA)
        register_parser(ParserB)
        assert isinstance(get_parser("txt"), ParserB)
