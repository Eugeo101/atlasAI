"""
SQLAlchemy models for the real estate AI agent project.
Phase 0 scope: minimal columns only. Phase 2 will extend this
(embeddings refs, tool-call logs, etc.) — that's when Alembic gets introduced.
"""
import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from pathlib import Path
target_dir = Path(__file__).resolve().parents[2]
load_dotenv(target_dir / '.env.example')

DATABASE_URL = os.environ.get(
    "DATABASE_URL"
)

Base = declarative_base()


class Unit(Base):
    __tablename__ = "units"

    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=True)
    unit_type = Column(String(50), nullable=False)   # apartment / villa / studio / office
    price = Column(Float, nullable=False)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Integer, nullable=False)
    area_sqm = Column(Float, nullable=False)
    is_available = Column(Boolean, default=True)
    description = Column(Text, nullable=True)   # stand-in for brochure text, Phase 2 will chunk this
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)   # user / assistant / system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine():
    return create_engine(DATABASE_URL, echo=False)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Tables created: {list(Base.metadata.tables.keys())}")
