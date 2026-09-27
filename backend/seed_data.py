"""Seed script: (re)creates claims.db and fills it with fake demo data.

Run directly:
    python3 seed_data.py

This is the ONLY script that should write to the database at import/run
time - db.py, tools.py and main.py stay side-effect-free on import.
"""

from db import get_connection, init_db, DB_PATH

CLAIMS = [
    # id,          client,          amount,   status,      type,     description
    ("CLM-1001", "Ivan Petrov", 12000.0, "pending", "Auto", "Rear-end collision on highway, bumper and trunk damage."),
    ("CLM-1002", "Maria Ivanova", 3200.0, "approved", "Health", "Emergency room visit for a broken wrist."),
    ("CLM-1003", "Sergei Volkov", 18500.0, "pending", "Home", "Kitchen fire damage, claim for appliances and cabinetry."),
    ("CLM-1004", "Olga Smirnova", 1500.0, "rejected", "Auto", "Minor parking lot scratch, no police report filed."),
    ("CLM-1005", "Dmitry Sokolov", 9800.0, "pending", "Health", "Physiotherapy sessions after a workplace injury."),
    ("CLM-1006", "Anna Kuznetsova", 19500.0, "pending", "Home", "Pipe burst flooding basement, requesting full renovation cost."),
    ("CLM-1007", "Ivan Petrov", 4500.0, "approved", "Auto", "Windshield replacement after storm damage."),
]

CUSTOMER_HISTORY = [
    # client,           claim_id,   date,         outcome
    ("Ivan Petrov", "CLM-0891", "2023-02-10", "approved"),
    ("Ivan Petrov", "CLM-0950", "2023-11-05", "fraud_detected"),
    ("Ivan Petrov", "CLM-1007", "2024-06-01", "approved"),
    ("Maria Ivanova", "CLM-0777", "2022-09-18", "approved"),
    ("Maria Ivanova", "CLM-1002", "2024-01-20", "approved"),
    ("Sergei Volkov", "CLM-0654", "2021-05-30", "rejected"),
    ("Sergei Volkov", "CLM-0910", "2023-08-14", "approved"),
    ("Olga Smirnova", "CLM-1004", "2024-03-02", "rejected"),
    ("Dmitry Sokolov", "CLM-0733", "2022-12-01", "approved"),
    ("Anna Kuznetsova", "CLM-0888", "2023-04-22", "approved"),
]


def _insert_seed_rows() -> None:
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO claims (id, client, amount, status, type, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            CLAIMS,
        )
        conn.executemany(
            "INSERT INTO customer_history (client, claim_id, date, outcome) "
            "VALUES (?, ?, ?, ?)",
            CUSTOMER_HISTORY,
        )
        conn.commit()
    finally:
        conn.close()


def seed() -> None:
    """Force a fresh reseed: wipes claims.db and recreates it from scratch."""
    if DB_PATH.exists():
        DB_PATH.unlink()

    init_db()
    _insert_seed_rows()

    print(f"Seeded {len(CLAIMS)} claims and {len(CUSTOMER_HISTORY)} customer_history rows into {DB_PATH}")


def ensure_seeded() -> None:
    """Create tables if missing and seed demo data only if claims is empty.

    Used from the FastAPI startup hook so `uvicorn main:app` works out of the
    box without a manual `python3 seed_data.py` step, while not clobbering
    data (e.g. status changes) on every --reload restart.
    """
    init_db()
    conn = get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    finally:
        conn.close()

    if count == 0:
        _insert_seed_rows()


if __name__ == "__main__":
    seed()
