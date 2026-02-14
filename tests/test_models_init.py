"""测试 models 包的公共导出"""


def test_orm_exports():
    """验证 ORM 模型可从 app.models 直接导入"""
    from app.models import Base, Chunk, Document, Entity, Extraction, Setting, Template

    assert Base is not None
    assert Document.__tablename__ == "documents"
    assert Chunk.__tablename__ == "chunks"
    assert Entity.__tablename__ == "entities"
    assert Template.__tablename__ == "templates"
    assert Extraction.__tablename__ == "extractions"
    assert Setting.__tablename__ == "settings"


def test_database_exports():
    """验证数据库工具函数可从 app.models 直接导入"""
    import inspect

    from app.models import close_db, get_db, init_db

    assert get_db is not None
    assert inspect.iscoroutinefunction(init_db)
    assert inspect.iscoroutinefunction(close_db)
