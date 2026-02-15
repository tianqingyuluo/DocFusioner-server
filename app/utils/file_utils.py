"""
文件工具函数。

上传存储、内容去重、MIME 识别、路径规范化、安全删除。
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
import zipfile
from datetime import date
from pathlib import Path
from typing import IO, BinaryIO

from app.config import get_settings
from app.exceptions import FileDeleteError, FileTooLargeError, UnsupportedFileTypeError

MAX_UPLOAD_SIZE: int = 30 * 1024 * 1024

_EXT_MAP: dict[str, str] = {
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".md": "md",
    ".txt": "txt",
    ".pdf": "pdf",
}

_ZIP_INTERNAL: dict[str, str] = {
    "docx": "word/document.xml",
    "xlsx": "xl/workbook.xml",
}

_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"

CHUNK_SIZE = 64 * 1024


def sanitize_filename(name: str) -> str:
    """
    文件名安全化。

    保留：中文、字母、数字、-_.
    去除：其他特殊字符
    """
    cleaned = re.sub(r"[^\u4e00-\u9fff\w.\-]", "", name)
    cleaned = cleaned.strip(" .")
    if not cleaned:
        return "upload"
    if len(cleaned) > 80:
        cleaned = cleaned[:80]
    return cleaned


def detect_file_type(filename: str) -> str:
    """
    扩展名 → FileType（白名单制）。

    不在白名单内抛 UnsupportedFileTypeError。
    """
    ext = Path(filename).suffix.lower()
    if ext not in _EXT_MAP:
        ext_display = ext.lstrip(".") if ext else ""
        raise UnsupportedFileTypeError(
            message=f"不支持的文件类型: {ext_display or filename}",
            file_type=ext_display or filename,
        )
    return _EXT_MAP[ext]


def validate_magic_bytes(file_path: Path, expected_type: str) -> bool:
    """
    magic bytes 校验。

    - txt/md：纯文本，始终返回 True
    - pdf：检查 %PDF- 头
    - docx/xlsx：检查 ZIP 头 + 内部文件存在性
    """
    if expected_type in ("txt", "md"):
        return True

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
    except OSError:
        return False

    if expected_type == "pdf":
        return header[:5] == _PDF_MAGIC

    if expected_type in _ZIP_INTERNAL:
        if header[:4] != _ZIP_MAGIC:
            return False
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                return _ZIP_INTERNAL[expected_type] in zf.namelist()
        except (zipfile.BadZipFile, OSError):
            return False

    return False


def compute_hash(file_like: BinaryIO | IO[bytes]) -> str:
    """流式计算 SHA-256 哈希。"""
    h = hashlib.sha256()
    while True:
        chunk = file_like.read(CHUNK_SIZE)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()


def get_upload_path(filename: str) -> Path:
    """
    生成上传文件存储路径。

    格式：{upload_dir}/{YYYY-MM-DD}/{short_uuid}_{safe_filename}
    """
    settings = get_settings()
    safe_name = sanitize_filename(filename)
    short_uuid = uuid.uuid4().hex[:8]
    today = date.today().isoformat()
    return Path(settings.upload_dir) / today / f"{short_uuid}_{safe_name}"


async def save_upload(upload) -> tuple[Path, str, int]:
    """
    保存上传文件（原子落盘）。

    返回 (file_path, content_hash, file_size)
    """
    final_path = get_upload_path(upload.filename)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256()
    total_size = 0

    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise FileTooLargeError(
                        message=f"文件 {upload.filename} 超过最大限制 {MAX_UPLOAD_SIZE} 字节",
                        filename=upload.filename,
                        size=total_size,
                        max_size=MAX_UPLOAD_SIZE,
                    )
                f.write(chunk)
                h.update(chunk)
        os.replace(tmp_path, final_path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return final_path, h.hexdigest(), total_size


def delete_file(file_path: Path) -> None:
    """
    幂等删除文件。

    - 文件不存在 → 静默忽略
    - 权限/IO 错误 → 抛 FileDeleteError
    """
    try:
        file_path.unlink(missing_ok=True)
    except PermissionError as e:
        raise FileDeleteError(
            message=f"删除文件失败: {file_path}",
            path=str(file_path),
            reason=str(e),
        ) from e
    except OSError as e:
        raise FileDeleteError(
            message=f"删除文件失败: {file_path}",
            path=str(file_path),
            reason=str(e),
        ) from e
