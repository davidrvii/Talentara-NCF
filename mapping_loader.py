import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

# Create connection to database
def get_db_connection():
    connection = pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "talentara"),
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

# Load mapping from DB (table, name_col, id_col)
def load_mapping_from_db(table, name_col, id_col):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        sql = f"SELECT {id_col}, {name_col} FROM {table} ORDER BY {id_col}"
        cursor.execute(sql)
        rows = cursor.fetchall()

    conn.close()

    # For mapping: { name : id }
    mapping = { row[name_col]: row[id_col] for row in rows }
    return mapping
