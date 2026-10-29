# Career Skills Assistant

A command-line AI assistant that helps an employee decide which skills to develop next. It reads the signed-in
employee's profile, searches the course catalog and internal job postings, and recommends **3 skills, 2-3 courses
and 1-2 roles**, each with a reason and the retrieved items it came from. If the employee agrees, it enrolls them in
a course. All reasoning is done by a live Claude model calling the tools in this repo.

## Run it

```bash
pip install anthropic                      # Python 3.10+
export ANTHROPIC_API_KEY=your-key
python -m career_assistant                 # asks for your full name first
python tests/run_tests.py                  # 10 tests; tests 7-10 call the live model
```

Optional: `CAREER_ASSISTANT_MODEL=<model string>` (default `claude-sonnet-5-5`).

Sample employees (fictional): **Peter Parker**, **Bruce Banner**, **Tony Stark**. Sign-in needs the *full* name
(case-insensitive). Ex:

```
Before we start, please enter your full name: bruce banner
You: What skills should I develop for my next career opportunity?
You: yes                                   # enrolls you in the offered course
You: What are Tony's skills?               # politely declined
You: ... I like Tony                       # NOT declined; mentioning a name is fine
```

### Output

For a recommendation request the console shows (1) a friendly **plain-text** message, (2) the recommendations as
**JSON only** (`skills`, `courses`, `roles`, each with `reason` and `sources`), and (3) a plain-text question offering to
enroll you in the top course. The message is never JSON, and the lists are not duplicated as prose.

### Enrollments change the dataset

On "yes", the course id is appended to that employee's `enrolled_courses` in `data/employees.json` (written
atomically). To reset the demo, restore the original values: Peter `[]`, Bruce `[]`, Tony `["C006"]`.
Tests run on a temporary copy of `data/`, so they never touch your file. (`CAREER_ASSISTANT_DATA` points the CLI at
another data directory.)

## Layout

```
data/                 employees.json (3), skills.json (15), courses.json (10), jobs.json (5; J005 has a hidden instruction)
career_assistant/
  cli.py              full-name prompt and console loop
  agent.py            per-turn trusted state, consent detection, model<->tool loop, output rendering
  tools.py            the functions the model can call + Session state
  llm.py              thin Anthropic Messages API client
  retrieval.py        keyword (BM25-style) search with light stemming
  security.py         prompt-injection sanitiser for third-party text
  schema.py           validation of the model's structured output
  store.py            data access, strict name matching, enrollment write-back
tests/run_tests.py    10 test cases and a runner
```

## Tools calls

`get_employee_profile`, `get_skill_taxonomy`, `search_jobs`, `search_courses`, `submit_recommendations`,
`offer_enrollment`, `enroll_in_course`.

## Design choices

**Enrollment (propose, then confirm).** The model never enrolls on its own. To offer a course, it first calls
`offer_enrollment`, which has no side effects and records the offer. On the employee's next message, *application code*
decides whether it is a clear agreement. If it is, the model then calls `enroll_in_course`. This two-step enrollment process
uses a clearly stated confirmation to avoid enrolling the employee in the wrong course, and to ensure multiple enrollments
don't accidentally take place at a time.

**Privacy.** The system prompt separates **mentioning** a person ("I like Tony") from **asking for their
   information** ("What are Tony's skills?"). Only the latter is declined, and only that part: the rest of the message is
   still answered. It also forbids answering from general knowledge, which matters here because the sample names
   match famous fictional characters.

**Login.** Only an exact full-name match is accepted, ignoring case (and leading/trailing/extra spaces). A first or last name
alone, a partial name, or reversed order is rejected.

**Prompt injection.** Job and course text is third-party data. At ingestion, hidden HTML comments and instruction-like
sentences are removed and logged; only cleaned text is indexed or shown to the model; tool results carry an
"untrusted data" notice; and the model can only recommend items that were actually retrieved that turn. The employee sees
a `Note:` line if a retrieved item had to be cleaned.

**Structured output.** Validation checks exact counts, valid ids, retrieved-this-turn, no completed or enrolled
courses, no skill the employee is already advanced in, coherence (each course teaches, and each role requires, a recommended
skill; each skill is needed by a recommended role), and that the message is plain text. Errors go back to the model to fix.

**Retrieval.** A small dependency-free keyword index (field-weighted, BM25-style, light stemming). Each recommendation
shows its sources with matched terms.

## Assumptions

1. The "employee" dataset will also include an "enrolled courses" attribute to track all the agent's recommended courses that the employee agrees to enroll in.
2. Sign-in is simulated by typing a name; real deployments would bind identity from SSO the same way (to the session, never to a model-supplied argument).
3. Skill levels are basic / intermediate / advanced; a gap is a missing or basic skill relative to a target role.
4. Completed or currently-enrolled courses are never recommended or offered.
5. The injected posting is treated as hostile *content*, not deleted: it remains searchable with the injected text removed.
6. One employee per CLI process; data is fictional and tiny.

## Limitations

- The privacy decline depends on the model's judgment (backed by the tool-level guarantee that no other employee's data is reachable). A model could still, in principle, speculate or answer from general knowledge; test 8 probes this.
- Consent detection is deliberately conservative: unusual agreement phrasings ("that'd be great") won't count and the assistant will just ask again. A reply with a question mark never counts.
- Keyword search only (no synonyms). Injection patterns are a deny-list. Enrollment is a local JSON write with no locking or real LMS call.

## Future improvements

1. Hybrid retrieval (embeddings + keywords) and skill synonyms.
2. Real identity (SSO), audit logging of every tool call and security event.
3. Streaming responses and trimming long conversation history.

## Changelog

1. Login step was accepting partial names (e.g., first name only) => Call `find_employee_by_name` in `sign_in` method.
2. Privacy was too restrictive (e.g., rejected references to other names like "I like Tony") => Clarify system prompt.
