from configurations import collection
from models.userModel import User
from schemas.userSchema import all_users
from bson.objectid import ObjectId
from datetime import datetime
from fastapi import HTTPException

# Get all users
async def get_all_users():
    data = collection.find({"is_deleted": False})
    return all_users(data)

# Create a new user
async def create_user(new_user: dict):
    try:
        resp = collection.insert_one(new_user)
        return {"status_code": 200, "id": str(resp.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Some error occurred: {e}")

# Update an existing user
async def update_user(user_id: str, updated_user: dict):
    try:
        id = ObjectId(user_id)
        existing_doc = collection.find_one({"_id": id, "is_deleted": False})
        if not existing_doc:
            raise HTTPException(status_code=404, detail="User does not exist")

        updated_user["updated_at"] = int(datetime.timestamp(datetime.now()))
        resp = collection.update_one({"_id": id}, {"$set": updated_user})
        return {"status_code": 200, "message": "User Updated Successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Some error occurred: {e}")

# Delete a user
async def delete_user(user_id: str):
    try:
        id = ObjectId(user_id)
        existing_doc = collection.find_one({"_id": id, "is_deleted": False})
        if not existing_doc:
            raise HTTPException(status_code=404, detail="User does not exist")

        resp = collection.update_one({"_id": id}, {"$set": {"is_deleted": True}})
        return {"status_code": 200, "message": "User Deleted Successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Some error occurred: {e}")
