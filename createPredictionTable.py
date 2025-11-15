import psycopg2

DATABASE_URL = "postgresql://assignment11modellogs_user:MQm5DZXgI9Z0J7TkYJreKY8GreBawbQo@dpg-d4cdrpjipnbc739di4gg-a.ohio-postgres.render.com/assignment11modellogs"

# Connect to the database
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# SQL statement to create table
create_table_query = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    M1 NUMERIC,
    M2 NUMERIC,
    M3 NUMERIC,
    M4 NUMERIC,
    M5 NUMERIC,
    M6 NUMERIC,
    Prediction NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Execute the query
cursor.execute(create_table_query)
conn.commit()

# Close the connection
cursor.close()
conn.close()