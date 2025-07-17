from shared import Database
from config import config

# Initialize a single database instance
db = Database(config.DATABASE_PATH)

def get_database():
    """FastAPI dependency: returns the shared Database instance"""
    return db