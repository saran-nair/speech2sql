import sqlite3

conn = sqlite3.connect("app/db/mock.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT
);
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount FLOAT
);
""")

# Insert mock data
cursor.execute("INSERT INTO customers (id, name, country) VALUES (1, 'Alice', 'Germany')")
cursor.execute("INSERT INTO customers (id, name, country) VALUES (2, 'Bob', 'UK')")
cursor.execute("INSERT INTO orders (customer_id, order_date, total_amount) VALUES (1, '2024-01-10', 250.50)")
cursor.execute("INSERT INTO orders (customer_id, order_date, total_amount) VALUES (2, '2024-03-15', 100.00)")

conn.commit()
conn.close()
