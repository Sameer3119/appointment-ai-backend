from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# ---------------------------------------------------------------------------
# Database Configuration & SQLite Engine Setup
# ---------------------------------------------------------------------------
# Note on SQLite:
# 'check_same_thread: False' is essential for SQLite when used with FastAPI.
# By default, SQLite only allows the thread that created the connection to use it.
# Because FastAPI handles concurrent requests across a thread pool, disabling this
# restriction prevents 'SQLite objects created in a thread can only be used in that same thread' errors.
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Create the SQLAlchemy Engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # Set to True if SQL query debugging is desired
)

# Create SessionLocal class: each call to SessionLocal() creates a distinct DB session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative base class for all SQLAlchemy ORM models
Base = declarative_base()


# ---------------------------------------------------------------------------
# Dependency Injection for Database Sessions
# ---------------------------------------------------------------------------
def get_db():
    """
    FastAPI dependency that yields an independent database session per request.
    
    Ensures that the session is properly closed when the request finishes,
    preventing connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
