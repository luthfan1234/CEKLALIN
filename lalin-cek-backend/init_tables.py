# init_tables.py
import asyncio
from database import init_db

async def main():
    print("Initializing database tables...")
    await init_db()
    print("Tables created successfully! ✅")

if __name__ == "__main__":
    asyncio.run(main())