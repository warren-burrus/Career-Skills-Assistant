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

### What you see

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

## Tools the model can call

`get_employee_profile`, `get_skill_taxonomy`, `search_jobs`, `search_courses`, `submit_recommendations`,
`offer_enrollment`, `enroll_in_course`.

## Design choices

**Enrollment (propose, then confirm).** The model never enrolls on its own. To offer a course it calls
`offer_enrollment`, which has no side effects and records the offer. On the employee's next message, *application code*
decides whether it is a clear agreement (`is_affirmative`: short, no question mark, no hedge or negation, so
"yes", "Yes, please enroll me in that one" and "sure, go ahead" count, while "yes but tell me more first" does not).
If it is, the course ids from that offer become the session's one-turn `consent`, and the model is told so in the
**system prompt** (a trusted "SESSION STATE" block), not in a pretend "[system]" message inside the chat. Not every model
supports forcing a tool call (`tool_choice`), so the code instead *verifies*: if the model replies without calling
`enroll_in_course`, it retries once with a firmer reminder, and if that also fails it says so honestly and keeps the offer
open. The tool itself refuses anything not in `consent`: a course that wasn't offered, a second use, or any text the
employee types that merely claims confirmation. After
recommendations, the exact text the employee saw (including the offer) is added to the conversation history, so the
model knows what "yes" refers to.

**Privacy has three layers, and none of them blocks harmless mentions of a name.**
1. *Code:* no tool takes an employee identifier. `get_employee_profile` always returns the signed-in employee's own
   record, so the model cannot read anyone else's data even if it tries.
2. *Model instructions:* the system prompt separates **mentioning** a person ("I like Tony") from **asking for their
   information** ("What are Tony's skills?"). Only the latter is declined, and only that part: the rest of the message is
   still answered. It also forbids answering from general knowledge, which matters here because the sample names
   match famous fictional characters.
3. *Hint:* if the message contains another employee's first or last name, a privacy note is added to the model's trusted
   state as a reminder. It is a hint, never a block.

**Login.** Only an exact full-name match is accepted, ignoring case (and leading/trailing/extra spaces). A first or last name
alone, a partial name, or reversed order is rejected.

**Prompt injection.** Job and course text is third-party data. At ingestion, hidden HTML comments and instruction-like
sentences are removed and logged; only cleaned text is indexed or shown to the model; tool results carry an
"untrusted data" notice; and the model can only recommend items that were actually retrieved that turn. The employee sees
a `Note:` line if a retrieved item had to be cleaned.

**Structured output.** The model submits through `submit_recommendations`, supplying only ids/names, reasons and the
friendly message. The code adds titles and provenance (matched terms and scores from the real search hits), so
sources can't be invented. Validation checks exact counts, valid ids, retrieved-this-turn, no completed or enrolled
courses, no skill the employee is already advanced in, coherence (each course teaches, and each role requires, a recommended
skill; each skill is needed by a recommended role), and that the message is plain text. Errors go back to the model to fix. The
JSON is re-parsed strictly before display.

**Retrieval.** A small dependency-free keyword index (field-weighted, BM25-style, light stemming). Each recommendation
shows its sources with matched terms.

## Assumptions

1. Sign-in is simulated by typing a name; real deployments would bind identity from SSO the same way (to the session, never to a model-supplied argument).
2. Skill levels are basic / intermediate / advanced; a gap is a missing or basic skill relative to a target role.
3. Completed or currently-enrolled courses are never recommended or offered.
4. The injected posting is treated as hostile *content*, not deleted: it remains searchable with the injected text removed.
5. "The employee agrees" means a short, unhedged affirmative reply to an offer made in the assistant's previous message. An offer lapses after one reply.
6. One employee per CLI process; data is fictional and tiny.

## Known limitations

- The privacy decline depends on the model's judgment (backed by the tool-level guarantee that no other employee's data is reachable). A model could still, in principle, speculate or answer from general knowledge; test 8 probes this.
- Consent detection is deliberately conservative: unusual agreement phrasings ("that'd be great") won't count and the assistant will just ask again. A reply with a question mark never counts.
- Keyword search only (no synonyms). Injection patterns are a deny-list. Enrollment is a local JSON write with no locking or real LMS call.

## What I'd improve next

1. Run and tune against the live model; add a model-graded eval set for recommendation quality and refusals.
2. Hybrid retrieval (embeddings + keywords) and skill synonyms.
3. Real identity (SSO), audit logging of every tool call and security event.
4. A classifier-based privacy and injection screen in addition to the prompt and sanitiser.
5. Real LMS integration with idempotent enrollment, prerequisites, capacity and undo.
6. Streaming responses and trimming long conversation history.
