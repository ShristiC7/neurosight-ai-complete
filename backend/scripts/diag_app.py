"""
Minimal diagnostic: import the app and run the lifespan manually,
then make a request to /health. This isolates where the hang is.
"""
import os, asyncio, sys
sys.setrecursionlimit(5000)

os.environ["ENVIRONMENT"] = "testing"
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5433"
os.environ["POSTGRES_DB"] = "neurosight_test"
os.environ["POSTGRES_USER"] = "neurosight"
os.environ["POSTGRES_PASSWORD"] = "neurosight_dev_password"
os.environ["SECRET_KEY"] = "test-secret-key-exactly-32-characters-long!"
os.environ["JWT_SECRET_KEY"] = "test-jwt-key-exactly-32-characters-ok!"
os.environ["REDIS_HOST"] = "localhost"
os.environ["QDRANT_HOST"] = "localhost"

from app.core.config import get_settings
get_settings.cache_clear()

async def main():
    print("1. Importing app...")
    import fakeredis
    from app.core import redis as redis_mod
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    redis_mod.redis_client = client
    redis_mod.cache = redis_mod.CacheClient(client)
    print("2. Redis patched")

    from app.main import app
    print("3. App imported")

    from httpx import AsyncClient, ASGITransport
    print("4. Creating AsyncClient...")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        print("5. Making request to /health...")
        resp = await c.get("/health")
        print(f"6. Response: {resp.status_code} {resp.json()}")
    print("7. Done!")

if __name__ == "__main__":
    asyncio.run(main())
