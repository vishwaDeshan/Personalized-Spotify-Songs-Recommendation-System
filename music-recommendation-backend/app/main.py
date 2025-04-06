from fastapi import FastAPI, HTTPException
from routers import user_router
from pymongo.errors import ConnectionFailure
from configurations import client

app = FastAPI()

# Include the user router
app.include_router(user_router)

# Test the MongoDB connection
@app.get("/test-db-connection")
async def test_db_connection():
    try:
        # Attempt a simple ping to test the connection
        client.admin.command('ping')
        return {"status": "success", "message": "Database connection successful"}
    except ConnectionFailure as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")
