# from sqlalchemy import create_engine, inspect

# # Database connection details
# DB_NAME = "olist"
# DB_USERNAME = "postgres"
# DB_PASSWORD = "admin"
# DB_HOST = "localhost"
# DB_PORT = "5432"

# # Create the database connection string
# db_url = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# engine = create_engine(db_url)

# # Connect to the database and list all tables
# try:
#     inspector = inspect(engine)
#     tables = inspector.get_table_names()
#     print("Tables in the database:")
#     for table in tables:
#         print(table)
# except Exception as e:
#     print(f"An error occurred: {e}")


from sqlalchemy import create_engine

# Database connection details
DB_NAME = "olist"
DB_USERNAME = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create the database connection string
db_url = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(db_url)


try:
    # Test the connection
    engine = create_engine(db_url)
    with engine.connect() as connection:
        print("Connection successful")
except Exception as e:
    print(f"Failed to connect to the database: {e}")
