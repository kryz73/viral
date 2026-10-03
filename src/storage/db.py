from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

# 1. Database Connection URL (SQLite default, easily swappable with Postgres)
DATABASE_URL = "sqlite:///./data/viral.db"

# 2. Ensure the parent directory exists for SQLite
if DATABASE_URL.startswith("sqlite"):
    Path("data").mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

# 3. Create Engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
)

# 4. Create Session Factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# 5. Declarative Base for ORM Models
class Base(DeclarativeBase):
    pass

# 6. Session Context Manager / FastAPI Dependency
def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session scope."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 7. Table Initializer
def init_db() -> None:
    """Create all tables defined in models.py if they do not exist."""
    import src.storage.models  # noqa: F401
    Base.metadata.create_all(bind=engine)