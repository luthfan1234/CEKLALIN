from sqlalchemy import Column, Integer, String, Float, JSON, TIMESTAMP, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime

Base = declarative_base()

# --- Model Database (SQLAlchemy) ---
class StreamDB(Base):
    __tablename__ = 'streams'
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, nullable=False)
    status = Column(String, default='idle')
    roi_points = Column(JSON)
    real_width_m = Column(Float)
    real_height_m = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    events = relationship("EventDB", back_populates="stream")

class EventDB(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(Integer, ForeignKey('streams.id'))
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    event_type = Column(String, nullable=False)
    object_type = Column(String)
    details = Column(JSON)
    severity = Column(String, default='low')
    stream = relationship("StreamDB", back_populates="events")

# --- Model API (Pydantic) ---
# FIX: Nama field disamakan dengan model database (real_width_m, real_height_m)
class StreamBase(BaseModel):
    url: str
    roi_points: Dict[str, List[float]]
    real_width_m: float
    real_height_m: float

class StreamCreate(StreamBase):
    pass

class Stream(StreamBase):
    id: int
    status: str
    
    class Config:
        from_attributes = True

class Event(BaseModel):
    id: int
    stream_id: int
    timestamp: datetime
    event_type: str
    object_type: str | None
    details: Dict[str, Any]
    severity: str
    
    class Config:
        from_attributes = True
