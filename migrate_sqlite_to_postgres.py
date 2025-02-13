import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Connect to SQLite
sqlite_conn = sqlite3.connect("backend/instance/example.db")
cursor = sqlite_conn.cursor()

# PostgreSQL connection
pg_conn_str = "postgresql://neondb_owner:npg_GpF01WdNIYBM@ep-frosty-dream-a5ya4h7n-pooler.us-east-2.aws.neon.tech/neondb?options=project%3Dep-frosty-dream-a5ya4h7n-pooler"
pg_engine = create_engine(pg_conn_str)

# Define the migration order to satisfy foreign key constraints
migration_order = ["team", "player", "player_round_performance"]

# Migrate each table in the correct order
for table_name in migration_order:
    print(f"\n🔍 Checking table: {table_name}")

    # Read data from SQLite
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
    print(f"📊 Found {len(df)} rows in {table_name}")

    if not df.empty:
        df.to_sql(table_name, pg_engine, if_exists="append", index=False)
        print(f"✅ Migrated {len(df)} rows from {table_name} to PostgreSQL")
    else:
        print(f"⚠️ No data found in {table_name}, skipping...")

# Close connections
sqlite_conn.close()
print("\n🎉 Migration completed successfully.")
