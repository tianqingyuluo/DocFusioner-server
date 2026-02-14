"""
Pydantic 数据模型（API 层 Schemas）。

所有 API 请求/响应的类型定义。
ORM 模型 → Schema 转换通过 ConfigDict(from_attributes=True) 支持。
"""

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

FileType = Literal["docx", "xlsx", "md", "txt", "pdf"]
TemplateFileType = Literal["docx", "xlsx"]
EntityType = Literal["person", "org", "date", "amount", "location"]
FieldStatus = Literal["filled", "missing", "ambiguous", "error"]
TaskStatus = Literal["pending", "processing", "completed", "failed"]

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    code: int
    message: str
    detail: str | None = None
    request_id: str | None = None


class PaginationParams(BaseModel):
    """分页参数。"""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应。"""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class ProgressStatus(BaseModel):
    """通用任务进度。"""

    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0)
    message: str | None = None


class DocumentResponse(BaseModel):
    """文档响应（完整信息）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: FileType
    file_size: int | None
    title: str | None
    summary: str | None
    chunk_count: int
    status: TaskStatus
    created_at: datetime


class DocumentListItem(BaseModel):
    """文档列表项（精简信息）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: FileType
    file_size: int | None
    chunk_count: int
    status: TaskStatus
    created_at: datetime


class EntityResponse(BaseModel):
    """实体响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    entity_type: EntityType
    entity_value: str
    normalized_value: str
    confidence: float


class DocumentDetailResponse(DocumentResponse):
    """文档详情（含实体列表）。"""

    entities: list[EntityResponse] = Field(default_factory=list)


class ChatMessageRequest(BaseModel):
    """聊天消息请求。"""

    message: str = Field(min_length=1, max_length=4000)
    doc_ids: list[int] | None = Field(
        default=None,
        description="引用的文档 ID 列表。None=全库检索，空列表视为 None",
    )


class SourceReference(BaseModel):
    """来源引用。"""

    doc_id: int
    filename: str
    section: str | None
    chunk_index: int
    chunk_id: int | None = None
    snippet: str = Field(description="相关文本片段")


class ChatResponse(BaseModel):
    """聊天响应。"""

    reply: str
    sources: list[SourceReference] = Field(default_factory=list)
    intent: str | None = Field(
        default=None,
        description="识别的意图：extract/edit/query/fill",
    )


class TemplateUploadResponse(BaseModel):
    """模板上传响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: TemplateFileType
    field_count: int
    created_at: datetime


class FillStartRequest(BaseModel):
    """填写任务启动请求。"""

    doc_ids: list[int] | None = Field(
        default=None,
        description="限定检索范围的文档 ID 列表。None=全库",
    )


class FillStartResponse(BaseModel):
    """填写任务启动响应。"""

    task_id: str
    template_id: int
    total_fields: int


class FieldResult(BaseModel):
    """单个字段的填写结果。"""

    field_name: str
    field_value: str | None
    status: FieldStatus
    confidence: float
    source_doc: str | None = Field(default=None, description="来源文档名")
    source_section: str | None = Field(default=None, description="来源章节")
    evidence_snippet: str | None = Field(default=None, description="证据片段")


class FillResult(BaseModel):
    """填写任务结果。"""

    template_id: int
    template_filename: str
    total_fields: int
    filled_fields: int
    high_confidence: int = Field(description="置信度 >= 0.8")
    low_confidence: int = Field(description="置信度 < 0.5")
    fields: list[FieldResult] = Field(default_factory=list)


class FillProgressResponse(BaseModel):
    """填写任务进度。"""

    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0)
    current_field: str | None = None
    current_index: int | None = None
    completed_fields: int = 0
    total_fields: int = 0
