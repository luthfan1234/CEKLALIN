import os
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import text
from dotenv import load_dotenv
from models import StreamDB, EventDB, Stream, Event

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Fungsi untuk inisialisasi tabel (jalankan sekali saat setup)
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(StreamDB.metadata.create_all)
        await conn.run_sync(EventDB.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def create_stream(
    db: AsyncSession,
    url: str,
    roi_points: dict,
    real_width_m: float,
    real_height_m: float
) -> Stream:
    # cek apakah URL sudah ada
    result = await db.execute(select(StreamDB).where(StreamDB.url == url))
    existing = result.scalars().first()

    if existing:
        # update data lama
        existing.roi_points = roi_points
        existing.real_width_m = real_width_m
        existing.real_height_m = real_height_m
        await db.commit()
        await db.refresh(existing)
        return Stream.from_orm(existing)

    # kalau belum ada → buat data baru
    new_stream = StreamDB(
        url=url,
        roi_points=roi_points,
        real_width_m=real_width_m,
        real_height_m=real_height_m,
        status="idle"
    )
    db.add(new_stream)
    await db.commit()
    await db.refresh(new_stream)
    return Stream.from_orm(new_stream)


async def get_stream(db: AsyncSession, stream_id: int) -> Stream | None:
    result = await db.get(StreamDB, stream_id)
    return Stream.from_orm(result) if result else None

async def update_stream_status(db: AsyncSession, stream_id: int, status: str):
    stream = await db.get(StreamDB, stream_id)
    if stream:
        stream.status = status
        await db.commit()

async def create_event(db: AsyncSession, event_data: Dict[str, Any]) -> Event:
    new_event = EventDB(**event_data)
    db.add(new_event)
    await db.commit()
    await db.refresh(new_event)
    return Event.from_orm(new_event)

async def get_events_for_stream(db: AsyncSession, stream_id: int, limit: int = 50) -> List[Event]:
    query = text(f"SELECT * FROM events WHERE stream_id = :stream_id ORDER BY timestamp DESC LIMIT :limit")
    result = await db.execute(query, {"stream_id": stream_id, "limit": limit})
    events = result.fetchall()
    return [Event.from_orm(e) for e in events]