from bson import ObjectId

def individual_user(user):
    """
    Converts a single MongoDB user document to a dictionary.
    """
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "status": "active" if user["is_active"] else "inactive",
        "creation": user["creation"],
        "updated_at": user["updated_at"]
    }

def all_users(users):
    """
    Converts a list of MongoDB user documents to a list of dictionaries.
    """
    return [individual_user(user) for user in users]
