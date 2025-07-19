from config import config
import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')
from shared import Database

# Initialize database
db = Database(config.DATABASE_PATH)

# Dependency for database
def get_database():
    return db