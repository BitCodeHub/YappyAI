"""
Script to add the updated_at column to the conversations table
Run this once to update your database schema
"""
import os
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, DateTime
from datetime import datetime

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")

# Fix for Render PostgreSQL URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Connecting to database: {DATABASE_URL[:50]}...")

# Create engine
engine = sqlalchemy.create_engine(DATABASE_URL)

# Try to add the column
try:
    with engine.connect() as conn:
        # Add updated_at column if it doesn't exist
        conn.execute(sqlalchemy.text("""
            ALTER TABLE conversations 
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """))
        conn.commit()
        print("✅ Successfully added updated_at column to conversations table")
except Exception as e:
    print(f"❌ Error adding column: {e}")
    # Try alternative syntax for SQLite
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("""
                ALTER TABLE conversations 
                ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """))
            conn.commit()
            print("✅ Successfully added updated_at column (SQLite)")
    except Exception as e2:
        print(f"❌ SQLite error: {e2}")
        print("\nPlease add the column manually or recreate the table.")

print("\nDone!")