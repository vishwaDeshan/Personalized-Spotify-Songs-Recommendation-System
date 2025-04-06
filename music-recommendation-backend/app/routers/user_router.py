from fastapi import APIRouter, HTTPException
from services.user_service import get_all_users, create_user, update_user, delete_user

router = APIRouter()

@router.get("/")
async def get_all_users_route():
    return await get_all_users()

@router.post("/")
async def create_user_route(new_user: dict):
    return await create_user(new_user)

@router.put("/{user_id}")
async def update_user_route(user_id: str, updated_user: dict):
    return await update_user(user_id, updated_user)

@router.delete("/{user_id}")
async def delete_user_route(user_id: str):
    return await delete_user(user_id)
