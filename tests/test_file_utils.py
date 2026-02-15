"""测试文件工具函数。"""

import hashlib
import io
import os
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.exceptions import (
    FileDeleteError,
    FileMagicMismatchError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)


class TestSanitizeFilename:
    """文件名安全化测试"""

    def test_normal_filename(self):
        from app.utils.file_utils import sanitize_filename

        assert sanitize_filename("报告.docx") == "报告.docx"

    def test_preserves_chinese_and_alphanumeric(self):
        from app.utils.file_utils import sanitize_filename

        assert sanitize_filename("合同-2024_v2.pdf") == "合同-2024_v2.pdf"

    def test_removes_special_chars(self):
        from app.utils.file_utils import sanitize_filename

        result = sanitize_filename("file@#$%^&.txt")
        assert result == "file.txt"

    def test_strips_leading_trailing_dots_and_spaces(self):
        from app.utils.file_utils import sanitize_filename

        assert sanitize_filename("  ..name.. ") == "name"

    def test_empty_after_sanitize_returns_upload(self):
        from app.utils.file_utils import sanitize_filename

        assert sanitize_filename("@#$%") == "upload"

    def test_truncates_at_80_chars(self):
        from app.utils.file_utils import sanitize_filename

        long_name = "a" * 100 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 80


class TestDetectFileType:
    """文件类型检测测试"""

    def test_docx(self):
        from app.utils.file_utils import detect_file_type

        assert detect_file_type("report.docx") == "docx"

    def test_xlsx(self):
        from app.utils.file_utils import detect_file_type

        assert detect_file_type("data.xlsx") == "xlsx"

    def test_md(self):
        from app.utils.file_utils import detect_file_type

        assert detect_file_type("README.md") == "md"

    def test_txt(self):
        from app.utils.file_utils import detect_file_type

        assert detect_file_type("notes.txt") == "txt"

    def test_pdf(self):
        from app.utils.file_utils import detect_file_type

        assert detect_file_type("paper.pdf") == "pdf"

    def test_unsupported_raises(self):
        from app.utils.file_utils import detect_file_type

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            detect_file_type("image.png")
        assert exc_info.value.file_type == "png"

    def test_no_extension_raises(self):
        from app.utils.file_utils import detect_file_type

        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type("noextension")

    def test_case_insensitive(self):
        from app.utils.file_utils import detect_file_type

        assert detect_file_type("report.DOCX") == "docx"
        assert detect_file_type("data.Xlsx") == "xlsx"


class TestValidateMagicBytes:
    """magic bytes 校验测试"""

    def test_valid_pdf(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 rest of content")
        assert validate_magic_bytes(pdf_file, "pdf") is True

    def test_invalid_pdf(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"this is not a PDF")
        assert validate_magic_bytes(fake_pdf, "pdf") is False

    def test_valid_docx(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        docx_file = tmp_path / "test.docx"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<xml/>")
        docx_file.write_bytes(buf.getvalue())
        assert validate_magic_bytes(docx_file, "docx") is True

    def test_invalid_docx_missing_internal(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        fake_docx = tmp_path / "fake.docx"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("something_else.txt", "data")
        fake_docx.write_bytes(buf.getvalue())
        assert validate_magic_bytes(fake_docx, "docx") is False

    def test_valid_xlsx(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        xlsx_file = tmp_path / "test.xlsx"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("xl/workbook.xml", "<xml/>")
        xlsx_file.write_bytes(buf.getvalue())
        assert validate_magic_bytes(xlsx_file, "xlsx") is True

    def test_txt_always_true(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        assert validate_magic_bytes(txt_file, "txt") is True

    def test_md_always_true(self, tmp_path: Path):
        from app.utils.file_utils import validate_magic_bytes

        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello")
        assert validate_magic_bytes(md_file, "md") is True


class TestComputeHash:
    """流式 SHA-256 测试"""

    def test_compute_hash_matches_hashlib(self):
        from app.utils.file_utils import compute_hash

        data = b"hello world" * 1000
        expected = hashlib.sha256(data).hexdigest()
        result = compute_hash(io.BytesIO(data))
        assert result == expected

    def test_compute_hash_empty(self):
        from app.utils.file_utils import compute_hash

        expected = hashlib.sha256(b"").hexdigest()
        assert compute_hash(io.BytesIO(b"")) == expected


class TestGetUploadPath:
    """上传路径生成测试"""

    def test_path_contains_date_and_uuid(self):
        from app.utils.file_utils import get_upload_path

        with patch("app.utils.file_utils.get_settings") as mock_settings:
            mock_settings.return_value.upload_dir = "data/uploads"
            path = get_upload_path("合同.docx")

        assert "data/uploads" in str(path)
        parts = path.parts
        assert any(part.count("-") == 2 for part in parts)
        assert "_" in path.name
        assert path.name.endswith("合同.docx") or "合同" in path.name


class TestSaveUpload:
    """save_upload 测试"""

    @pytest.mark.asyncio
    async def test_save_upload_returns_path_hash_size(self, tmp_path: Path):
        from app.utils.file_utils import save_upload

        content = b"test file content"
        upload = MagicMock()
        upload.filename = "test.txt"
        upload.read = AsyncMock(side_effect=[content, b""])

        with patch("app.utils.file_utils.get_upload_path") as mock_path:
            final_path = tmp_path / "test.txt"
            mock_path.return_value = final_path
            path, content_hash, size = await save_upload(upload)

        assert path == final_path
        assert content_hash == hashlib.sha256(content).hexdigest()
        assert size == len(content)
        assert final_path.exists()

    @pytest.mark.asyncio
    async def test_save_upload_atomic_write(self, tmp_path: Path):
        """验证使用 .tmp 中间文件 + os.replace 原子写入"""
        from app.utils.file_utils import save_upload

        content = b"atomic test"
        upload = MagicMock()
        upload.filename = "atomic.txt"
        upload.read = AsyncMock(side_effect=[content, b""])

        final_path = tmp_path / "atomic.txt"
        with patch("app.utils.file_utils.get_upload_path", return_value=final_path):
            await save_upload(upload)

        assert not (tmp_path / "atomic.txt.tmp").exists()
        assert final_path.exists()

    @pytest.mark.asyncio
    async def test_save_upload_too_large_raises(self, tmp_path: Path):
        from app.utils.file_utils import save_upload

        chunk = b"x" * 1024
        chunks = [chunk] * 100
        chunks.append(b"")
        upload = MagicMock()
        upload.filename = "big.txt"
        upload.read = AsyncMock(side_effect=chunks)

        final_path = tmp_path / "big.txt"
        with (
            patch("app.utils.file_utils.get_upload_path", return_value=final_path),
            patch("app.utils.file_utils.MAX_UPLOAD_SIZE", 50 * 1024),
        ):
            with pytest.raises(FileTooLargeError) as exc_info:
                await save_upload(upload)

        assert exc_info.value.filename == "big.txt"
        assert not (tmp_path / "big.txt.tmp").exists()

    @pytest.mark.asyncio
    async def test_save_upload_creates_parent_dirs(self, tmp_path: Path):
        from app.utils.file_utils import save_upload

        upload = MagicMock()
        upload.filename = "deep.txt"
        upload.read = AsyncMock(side_effect=[b"data", b""])

        nested_path = tmp_path / "2026-02-15" / "deep.txt"
        with patch("app.utils.file_utils.get_upload_path", return_value=nested_path):
            await save_upload(upload)

        assert nested_path.exists()


class TestDeleteFile:
    """文件删除测试"""

    def test_delete_existing_file(self, tmp_path: Path):
        from app.utils.file_utils import delete_file

        f = tmp_path / "to_delete.txt"
        f.write_text("delete me")
        delete_file(f)
        assert not f.exists()

    def test_delete_nonexistent_is_idempotent(self, tmp_path: Path):
        from app.utils.file_utils import delete_file

        f = tmp_path / "nonexistent.txt"
        delete_file(f)

    def test_delete_permission_error_raises(self, tmp_path: Path):
        from app.utils.file_utils import delete_file

        f = tmp_path / "protected.txt"
        f.write_text("protected")

        with patch("pathlib.Path.unlink", side_effect=PermissionError("拒绝访问")):
            with pytest.raises(FileDeleteError) as exc_info:
                delete_file(f)
            assert "拒绝访问" in exc_info.value.reason
