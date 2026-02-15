"""测试统一异常模块。"""

from app.exceptions import (
    AppError,
    ChromaError,
    ChromaUpsertError,
    FileDeleteError,
    FileError,
    FileMagicMismatchError,
    FileTooLargeError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMOutputParseError,
    LLMRateLimitError,
    ParseError,
    UnsupportedFileTypeError,
)


class TestAppErrorBase:
    """AppError 基类测试"""

    def test_str_returns_message(self):
        err = AppError(message="出错了")
        assert str(err) == "出错了"

    def test_is_exception(self):
        assert issubclass(AppError, Exception)


class TestLLMErrors:
    """LLM 异常层级测试"""

    def test_llm_auth_error_inherits_llm_error(self):
        err = LLMAuthError(message="key无效", provider="deepseek", model="chat")
        assert isinstance(err, LLMError)
        assert isinstance(err, AppError)
        assert err.provider == "deepseek"
        assert err.model == "chat"

    def test_llm_connection_error_fields(self):
        err = LLMConnectionError(
            message="连接超时", provider="ollama", base_url="http://localhost:11434"
        )
        assert err.provider == "ollama"
        assert err.base_url == "http://localhost:11434"

    def test_llm_model_not_found_error_fields(self):
        err = LLMModelNotFoundError(message="模型不存在", provider="deepseek", model="gpt-5")
        assert err.provider == "deepseek"
        assert err.model == "gpt-5"

    def test_llm_output_parse_error_fields(self):
        err = LLMOutputParseError(
            message="JSON无效",
            reason="json_invalid",
            provider="deepseek",
            model="chat",
            raw_output='{"broken',
        )
        assert err.reason == "json_invalid"
        assert err.raw_output == '{"broken'

    def test_llm_rate_limit_error_fields(self):
        err = LLMRateLimitError(message="限流", provider="deepseek", retry_count=2)
        assert err.retry_count == 2


class TestFileErrors:
    """文件异常层级测试"""

    def test_unsupported_file_type_inherits_file_error(self):
        err = UnsupportedFileTypeError(message="不支持的类型", file_type="exe")
        assert isinstance(err, FileError)
        assert isinstance(err, AppError)
        assert err.file_type == "exe"

    def test_file_too_large_fields(self):
        err = FileTooLargeError(
            message="文件过大", filename="big.pdf", size=50_000_000, max_size=30_000_000
        )
        assert err.filename == "big.pdf"
        assert err.size == 50_000_000

    def test_file_delete_error_fields(self):
        err = FileDeleteError(message="删除失败", path="/tmp/a.txt", reason="权限不足")
        assert err.path == "/tmp/a.txt"

    def test_file_magic_mismatch_fields(self):
        err = FileMagicMismatchError(
            message="魔数不匹配", filename="fake.docx", expected="docx", actual_header="PNG"
        )
        assert err.expected == "docx"


class TestChromaErrors:
    """向量库异常层级测试"""

    def test_chroma_upsert_error_inherits(self):
        err = ChromaUpsertError(message="写入失败", failed_ids=["1_0", "1_1"], total=10)
        assert isinstance(err, ChromaError)
        assert isinstance(err, AppError)
        assert err.failed_ids == ["1_0", "1_1"]
        assert err.total == 10


class TestParseError:
    """解析异常测试"""

    def test_parse_error_fields(self):
        err = ParseError(message="文件损坏", filename="broken.pdf", reason="无法读取")
        assert isinstance(err, AppError)
        assert err.filename == "broken.pdf"
        assert err.reason == "无法读取"
