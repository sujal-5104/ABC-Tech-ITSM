
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=True)

# Read database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Validate required settings
required = {
    "DB_HOST": DB_HOST,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_NAME": DB_NAME,
}

missing = [key for key, value in required.items() if not value]
if missing:
    raise ValueError(
        f"Missing database settings in .env: {', '.join(missing)}"
    )

# Build the connection URL safely
url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    query={"connect_timeout": "10"},
)

# Create a limited connection pool
engine = create_engine(
    url,
    pool_pre_ping=True,
    pool_size=1,
    max_overflow=0,
    pool_recycle=1800,
    connect_args={
        "connect_timeout": 10,
        "read_timeout": 30,
    },
)


def test_connection():
    """Test the connection without modifying database records."""
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1")).scalar_one()
        print("Database connection successful:", result)


if __name__ == "__main__":
    test_connection()
