import psycopg2
import dotenv
import os

dotenv.load_dotenv()

DB_CONNECTION = os.environ.get("DB_CONNECTION")

with psycopg2.connect(DB_CONNECTION) as conn:
    with conn.cursor() as cur:
        with open("mental_health_vecs_export.csv", "w", newline="", encoding="utf-8") as f:
            cur.copy_expert("COPY vecs.mental_health TO STDOUT WITH (FORMAT csv, HEADER true)", f)