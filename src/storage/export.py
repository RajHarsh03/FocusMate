"""
FocusMate CSV export.

export_session_csv() writes all readings of one session to a CSV file, so
the data can be opened in Excel or used for the report.
"""

import csv
import os

try:
    from src.storage import database  # when imported from the project root
except ImportError:
    import database  # when run directly: python src/storage/export.py

# All columns of focus_readings, in table order. Used as the CSV header.
READING_COLUMNS = [
    "reading_id", "session_id", "timestamp", "elapsed_sec", "face_detected",
    "ear", "blinks_in_interval", "head_direction", "focus_preview",
]


def export_session_csv(session_id, output_path=None):
    """Write the readings of one session to CSV and return the file path.

    If output_path is None, the file goes to exports/session_<id>.csv next
    to the database file (data/exports/session_<id>.csv by default). Missing
    folders are created. Empty values (None) become empty cells. A session
    without readings gives a file with only the header row.
    Returns None if the session does not exist.
    """
    if database.get_session_summary(session_id) is None:
        return None

    if output_path is None:
        data_folder = os.path.dirname(database.DB_PATH)
        output_path = os.path.join(data_folder, "exports", f"session_{int(session_id)}.csv")

    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    readings = database.get_readings(session_id)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(READING_COLUMNS)
        for reading in readings:
            # csv.writer already writes None as an empty cell.
            writer.writerow([reading[column] for column in READING_COLUMNS])

    return output_path


# ---------------------------------------------------------------------------
# Small demo: python src/storage/export.py
# Exports the newest session to data/exports/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    database.init_db()
    latest = database.list_sessions(limit=1)
    if not latest:
        print("No sessions yet. Run: python src/storage/seed_demo_data.py")
    else:
        sid = latest[0]["session_id"]
        path = export_session_csv(sid)
        rows = len(database.get_readings(sid))
        print(f"Exported {rows} readings of session {sid} to {path}")
