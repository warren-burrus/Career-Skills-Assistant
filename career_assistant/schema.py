"""Validation of the structured recommendation output.

validate_payload returns human-readable errors (empty list = valid). Beyond shape it checks:
  * grounding: every recommended course/role was retrieved by a search tool THIS turn;
  * coherence: each course teaches, and each role requires, at least one recommended skill, and each
    recommended skill is required by at least one recommended role;
  * no course the employee already completed or is enrolled in; no skill they already have at advanced level;
  * the friendly message is plain text, never JSON.
Errors are returned to the model so it can fix and resubmit.
"""
from __future__ import annotations

import json


def parse_json_strict(text: str) -> dict:
    """json.loads that rejects NaN/Infinity and non-object results."""
    def _bad(const):
        raise ValueError(f"invalid JSON constant {const}")
    obj = json.loads(text, parse_constant=_bad)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    return obj


def _reason_error(path: str, item) -> str | None:
    r = item.get("reason") if isinstance(item, dict) else None
    if not isinstance(r, str) or not 15 <= len(r.strip()) <= 600:
        return f"{path}.reason must be a short explanation (15-600 characters)"
    return None


def message_errors(message) -> list[str]:
    if not isinstance(message, str) or len(message.strip()) < 20:
        return ["message must be a friendly plain-text message of at least 20 characters"]
    m = message.strip()
    if m[0] in "{[" or "```" in m:
        return ["message must be plain text, not JSON or code (the recommendations are shown as JSON separately)"]
    return []


def validate_payload(payload, store, employee: dict | None = None, retrieved: dict | None = None) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]
    errors = message_errors(payload.get("message"))
    skills, courses, roles = payload.get("skills"), payload.get("courses"), payload.get("roles")

    if not isinstance(skills, list) or len(skills) != 3:
        errors.append("skills must be a list of exactly 3 items")
    if not isinstance(courses, list) or not 2 <= len(courses) <= 3:
        errors.append("courses must be a list of 2 to 3 items")
    if not isinstance(roles, list) or not 1 <= len(roles) <= 2:
        errors.append("roles must be a list of 1 to 2 items")
    if errors:
        return errors   # cross-checks below need well-formed lists

    levels = {s["name"]: s["level"] for s in employee["skills"]} if employee else {}
    done = set(employee["completed_courses"]) | set(employee["enrolled_courses"]) if employee else set()
    retrieved = retrieved if retrieved is not None else None

    skill_names = []
    for i, s in enumerate(skills):
        name = s.get("skill") if isinstance(s, dict) else None
        if name not in store.skills:
            errors.append(f"skills[{i}].skill {name!r} is not in the skill taxonomy (use the exact taxonomy name)")
        elif levels.get(name) == "advanced":
            errors.append(f"skills[{i}] {name}: the employee is already advanced at this skill")
        skill_names.append(name)
        if (e := _reason_error(f"skills[{i}]", s)):
            errors.append(e)
    if len(set(skill_names)) != len(skill_names):
        errors.append("skills must be 3 distinct skills")
    rec = {n for n in skill_names if n in store.skills}

    role_skill_union: set[str] = set()
    seen = set()
    for i, r in enumerate(roles):
        jid = r.get("job_id") if isinstance(r, dict) else None
        if jid not in store.jobs:
            errors.append(f"roles[{i}].job_id {jid!r} does not exist")
        else:
            if jid in seen:
                errors.append(f"roles[{i}] repeats {jid}")
            seen.add(jid)
            required = set(store.jobs[jid]["required_skills"])
            role_skill_union |= required
            if retrieved is not None and jid not in retrieved["job"]:
                errors.append(f"roles[{i}] {jid} was not returned by search_jobs this turn; search first")
            if rec and not (required & rec):
                errors.append(f"roles[{i}] {jid} requires none of the 3 recommended skills")
        if (e := _reason_error(f"roles[{i}]", r)):
            errors.append(e)
    for n in rec:
        if n not in role_skill_union:
            errors.append(f"skill {n!r} is not required by any recommended role")

    seen = set()
    for i, c in enumerate(courses):
        cid = c.get("course_id") if isinstance(c, dict) else None
        if cid not in store.courses:
            errors.append(f"courses[{i}].course_id {cid!r} does not exist")
        else:
            if cid in seen:
                errors.append(f"courses[{i}] repeats {cid}")
            seen.add(cid)
            if cid in done:
                errors.append(f"courses[{i}] {cid}: the employee already completed it or is enrolled in it")
            if retrieved is not None and cid not in retrieved["course"]:
                errors.append(f"courses[{i}] {cid} was not returned by search_courses this turn; search first")
            if rec and not (set(store.courses[cid]["skills_taught"]) & rec):
                errors.append(f"courses[{i}] {cid} teaches none of the 3 recommended skills")
        if (e := _reason_error(f"courses[{i}]", c)):
            errors.append(e)
    return errors
