"""Data access layer for the JSON datasets.

Enrollments are written straight back into data/employees.json (the employee's
`enrolled_courses` list), using an atomic write so a crash can't corrupt the file.
Set CAREER_ASSISTANT_DATA to point at a different data directory (the tests use a temp copy).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data"


def normalize_name(name: str) -> str:
    """Case-insensitive, whitespace-tolerant form used for exact full-name matching."""
    return " ".join(name.split()).casefold()


class DataStore:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or os.environ.get("CAREER_ASSISTANT_DATA") or DEFAULT_DATA)
        self._employee_list = self._load("employees.json")
        self.employees = {e["id"]: e for e in self._employee_list}   # same dict objects as the list
        self.skills = {s["name"]: s for s in self._load("skills.json")}
        self.courses = {c["id"]: c for c in self._load("courses.json")}
        self.jobs = {j["id"]: j for j in self._load("jobs.json")}

    def _load(self, filename: str) -> list[dict]:
        with open(self.data_dir / filename, encoding="utf-8") as fh:
            return json.load(fh)

    # ---- employees ---------------------------------------------------------
    def find_employee_by_name(self, name: str) -> dict | None:
        """Exact FULL-name match, ignoring case (and extra whitespace). First or last name alone never matches."""
        wanted = normalize_name(name or "")
        if not wanted:
            return None
        for emp in self.employees.values():
            if normalize_name(emp["name"]) == wanted:
                return emp
        return None

    def profile(self, employee_id: str) -> dict:
        """A copy of the employee's current record."""
        return json.loads(json.dumps(self.employees[employee_id]))

    def other_employees(self, employee_id: str) -> list[dict]:
        return [e for i, e in self.employees.items() if i != employee_id]

    # ---- enrollment --------------------------------------------------------
    def enroll(self, employee_id: str, course_id: str) -> None:
        emp = self.employees[employee_id]
        if course_id not in emp["enrolled_courses"]:
            emp["enrolled_courses"].append(course_id)
        self._write_employees()

    def _write_employees(self) -> None:
        path = self.data_dir / "employees.json"
        fd, tmp = tempfile.mkstemp(dir=self.data_dir, prefix=".employees-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._employee_list, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
