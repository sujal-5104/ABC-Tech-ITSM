
from database import test_connection
from sqlalchemy.exc import SQLAlchemyError


if __name__ == "__main__":
    try:
        test_connection()
    except SQLAlchemyError as error:
        print("DATABASE CONNECTION FAILED")
        print("Error type:", type(error).__name__)
        print("Details:", error.orig)
