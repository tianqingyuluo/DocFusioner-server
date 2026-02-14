"""测试 Pydantic API Schemas"""

from datetime import datetime

import pytest
from pydantic import ValidationError


class TestErrorResponse:
    def test_basic(self):
        from app.models.schemas import ErrorResponse

        err = ErrorResponse(code=404, message="Not Found")
        assert err.code == 404
        assert err.detail is None


class TestPaginationParams:
    def test_defaults(self):
        from app.models.schemas import PaginationParams

        p = PaginationParams()
        assert p.page == 1
        assert p.page_size == 20

    def test_invalid_page(self):
        from app.models.schemas import PaginationParams

        with pytest.raises(ValidationError):
            PaginationParams(page=0)

    def test_page_size_limit(self):
        from app.models.schemas import PaginationParams

        with pytest.raises(ValidationError):
            PaginationParams(page_size=101)


class TestPaginatedResponse:
    def test_generic_int(self):
        from app.models.schemas import PaginatedResponse

        resp = PaginatedResponse[int](
            items=[1, 2, 3],
            total=100,
            page=1,
            page_size=20,
            total_pages=5,
        )
        assert len(resp.items) == 3
        assert resp.total_pages == 5


class TestDocumentResponse:
    def test_from_attributes(self):
        """验证 from_attributes=True 可从 ORM 对象构造"""
        from app.models.schemas import DocumentResponse

        data = {
            "id": 1,
            "filename": "test.docx",
            "file_type": "docx",
            "file_size": 1024,
            "title": None,
            "summary": None,
            "chunk_count": 5,
            "status": "completed",
            "created_at": datetime(2026, 1, 1),
        }
        resp = DocumentResponse(**data)
        assert resp.id == 1
        assert resp.status == "completed"

    def test_invalid_file_type(self):
        from app.models.schemas import DocumentResponse

        with pytest.raises(ValidationError):
            DocumentResponse(
                id=1,
                filename="x",
                file_type="invalid",
                file_size=0,
                title=None,
                summary=None,
                chunk_count=0,
                status="pending",
                created_at=datetime.now(),
            )

    def test_invalid_status(self):
        from app.models.schemas import DocumentResponse

        with pytest.raises(ValidationError):
            DocumentResponse(
                id=1,
                filename="x",
                file_type="docx",
                file_size=0,
                title=None,
                summary=None,
                chunk_count=0,
                status="invalid_status",
                created_at=datetime.now(),
            )


class TestDocumentDetailResponse:
    def test_inherits_document_response(self):
        from app.models.schemas import DocumentDetailResponse, DocumentResponse

        assert issubclass(DocumentDetailResponse, DocumentResponse)

    def test_with_entities(self):
        from app.models.schemas import DocumentDetailResponse, EntityResponse

        entity = EntityResponse(
            id=1,
            entity_type="person",
            entity_value="张三",
            normalized_value="张三",
            confidence=0.9,
        )
        detail = DocumentDetailResponse(
            id=1,
            filename="test.docx",
            file_type="docx",
            file_size=1024,
            title="标题",
            summary="摘要",
            chunk_count=5,
            status="completed",
            created_at=datetime.now(),
            entities=[entity],
        )
        assert len(detail.entities) == 1


class TestChatMessageRequest:
    def test_valid_message(self):
        from app.models.schemas import ChatMessageRequest

        req = ChatMessageRequest(message="你好")
        assert req.doc_ids is None

    def test_empty_message_rejected(self):
        from app.models.schemas import ChatMessageRequest

        with pytest.raises(ValidationError):
            ChatMessageRequest(message="")

    def test_message_too_long(self):
        from app.models.schemas import ChatMessageRequest

        with pytest.raises(ValidationError):
            ChatMessageRequest(message="x" * 4001)

    def test_with_doc_ids(self):
        from app.models.schemas import ChatMessageRequest

        req = ChatMessageRequest(message="查询", doc_ids=[1, 2, 3])
        assert req.doc_ids == [1, 2, 3]


class TestChatResponse:
    def test_basic(self):
        from app.models.schemas import ChatResponse

        resp = ChatResponse(reply="回答内容")
        assert resp.sources == []
        assert resp.intent is None


class TestFillModels:
    def test_fill_start_request(self):
        from app.models.schemas import FillStartRequest

        req = FillStartRequest()
        assert req.doc_ids is None

    def test_field_result(self):
        from app.models.schemas import FieldResult

        fr = FieldResult(
            field_name="项目名称",
            field_value="DocFusion",
            status="filled",
            confidence=0.95,
        )
        assert fr.source_doc is None
        assert fr.evidence_snippet is None

    def test_invalid_field_status(self):
        from app.models.schemas import FieldResult

        with pytest.raises(ValidationError):
            FieldResult(
                field_name="x",
                field_value="y",
                status="unknown",
                confidence=0.5,
            )

    def test_fill_result(self):
        from app.models.schemas import FillResult

        result = FillResult(
            template_id=1,
            template_filename="tpl.xlsx",
            total_fields=10,
            filled_fields=8,
            high_confidence=6,
            low_confidence=1,
        )
        assert result.fields == []

    def test_fill_progress_response(self):
        from app.models.schemas import FillProgressResponse

        prog = FillProgressResponse(
            task_id="task-123",
            status="processing",
            progress=0.5,
        )
        assert prog.completed_fields == 0

    def test_progress_out_of_range(self):
        from app.models.schemas import FillProgressResponse

        with pytest.raises(ValidationError):
            FillProgressResponse(
                task_id="x",
                status="processing",
                progress=1.5,
            )


class TestTemplateUploadResponse:
    def test_valid(self):
        from app.models.schemas import TemplateUploadResponse

        resp = TemplateUploadResponse(
            id=1,
            filename="tpl.docx",
            file_type="docx",
            field_count=5,
            created_at=datetime.now(),
        )
        assert resp.file_type == "docx"

    def test_invalid_template_type(self):
        from app.models.schemas import TemplateUploadResponse

        with pytest.raises(ValidationError):
            TemplateUploadResponse(
                id=1,
                filename="tpl.md",
                file_type="md",
                field_count=0,
                created_at=datetime.now(),
            )


class TestProgressStatus:
    def test_valid(self):
        from app.models.schemas import ProgressStatus

        ps = ProgressStatus(task_id="abc", status="pending", progress=0.0)
        assert ps.message is None

    def test_progress_boundaries(self):
        from app.models.schemas import ProgressStatus

        ProgressStatus(task_id="a", status="completed", progress=1.0)
        ProgressStatus(task_id="a", status="pending", progress=0.0)

        with pytest.raises(ValidationError):
            ProgressStatus(task_id="a", status="pending", progress=-0.1)
        with pytest.raises(ValidationError):
            ProgressStatus(task_id="a", status="pending", progress=1.1)
