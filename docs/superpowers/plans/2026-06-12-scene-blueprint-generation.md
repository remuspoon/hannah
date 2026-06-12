# Scene Blueprint Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add blueprint generation cells to `wukong/v2/hannah_memory_scene.ipynb` that take a single key event and produce a set of scene blueprint objects via the OpenAI API.

**Architecture:** Four new notebook cells are appended after the existing key events cells: a system prompt, a `create_blueprints()` function, a first-pass single-event run cell, and a batch-all-events cell. Each cell is self-contained and stores results in variables for inspection before saving.

**Tech Stack:** Python, OpenAI Python SDK (`client` already initialised in the notebook), `json`, `os`

---

### Task 1: Add `BLUEPRINT_SYSTEM_PROMPT` to the notebook

**Files:**
- Modify: `wukong/v2/hannah_memory_scene.ipynb` (insert after cell `990a72b4`)

- [ ] **Step 1: Insert a markdown section header cell after `990a72b4`**

Use NotebookEdit with `edit_mode: insert`, `cell_type: markdown`, `cell_id: 990a72b4`:

```markdown
## Blueprint Generation
```

- [ ] **Step 2: Read the notebook to get the new markdown cell's ID**

Use the Read tool on `wukong/v2/hannah_memory_scene.ipynb` and find the ID of the cell just inserted (it will be the cell immediately after `990a72b4`). Call it `<markdown_cell_id>`.

- [ ] **Step 3: Insert the system prompt code cell after `<markdown_cell_id>`**

Use NotebookEdit with `edit_mode: insert`, `cell_type: code`, `cell_id: <markdown_cell_id>`:

```python
BLUEPRINT_SYSTEM_PROMPT = """
You are a narrative scene architect. Your task is to generate a set of scene blueprints for one specific key event from Hannah's life.

## Context you receive
- target_event: the event for which you will generate blueprints
- all_events: the full chronological list of Hannah's key events — provided for continuity context ONLY

Use all_events to understand Hannah's emotional baseline before the target event and what follows after. Do NOT generate blueprints for any event other than target_event.

## Output format
Return a JSON object with exactly this shape:

{
  "blueprints": [
    {
      "id": "{event_id}_{time}_{interlocutor}",
      "time": "during",
      "interlocutor": "father",
      "location": "family home, hallway",
      "emotional_state": "confused, pleading",
      "description": "..."
    }
  ]
}

### Field rules

id — snake_case, format: {event_id}_{time}_{interlocutor}. Use the target event's id as prefix. No spaces.

time — One of exactly: before | during | immediately_after | long_after

interlocutor — The person Hannah is with in this scene. Infer from the target event — whoever is meaningfully present or involved. Use role names in snake_case (e.g. father, mother, the_bully, older_boy).

location — A specific, plausible location. Be precise (e.g. "family home, kitchen" not just "home"). Infer from event context and time period.

emotional_state — Hannah's emotional state in this scene. 2–4 words, specific (e.g. "numb, dissociated" — not just "sad").

description — 3–6 sentences, close third-person (refer to Hannah by name). Extend the target event's description to fit this blueprint's specific time, location, interlocutor, and emotional state.
  - Ground every fact in the autobiography — never contradict anything stated
  - You may add: sensory detail, atmosphere, Hannah's internal state, minor contextual specifics consistent with the text
  - Do NOT invent new characters or change event outcomes
  - Carry Hannah's cumulative emotional weight forward from prior events in all_events

## Permutation strategy

Generate one blueprint per combination of time × interlocutor. Infer all interlocutors meaningfully involved in the target event.

Time positions: before, during, immediately_after, long_after

Exception — single interlocutor events: if the event has only one interlocutor, generate two blueprints for the long_after slot with contrasting emotional states. Differentiate their ids with a suffix: {event_id}_long_after_{interlocutor}_numb and {event_id}_long_after_{interlocutor}_angry (or whatever the contrasting states are).

Target 4–8 blueprints total per event.

## Hard constraints
- Output only the JSON object. No markdown fences, no commentary, no preamble.
- Every blueprint must be a distinct scene — no two should feel like the same moment.
- long_after blueprints must reflect Hannah's cumulative experience across her life, not just the isolated event.
"""
```

- [ ] **Step 4: Commit**

```bash
git add wukong/v2/hannah_memory_scene.ipynb
git commit -m "add blueprint system prompt to memory scene notebook"
```

---

### Task 2: Add `create_blueprints()` function

**Files:**
- Modify: `wukong/v2/hannah_memory_scene.ipynb` (insert after the system prompt cell from Task 1)

- [ ] **Step 1: Read the notebook to get the system prompt cell's ID**

Use the Read tool on `wukong/v2/hannah_memory_scene.ipynb`. Find the cell containing `BLUEPRINT_SYSTEM_PROMPT = """`. Call its ID `<sysprompt_cell_id>`.

- [ ] **Step 2: Insert the function cell after `<sysprompt_cell_id>`**

Use NotebookEdit with `edit_mode: insert`, `cell_type: code`, `cell_id: <sysprompt_cell_id>`:

```python
def create_blueprints(target_event: dict, all_events: list) -> dict:
    payload = json.dumps(
        {"target_event": target_event, "all_events": all_events},
        ensure_ascii=False,
        indent=2
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": BLUEPRINT_SYSTEM_PROMPT},
            {"role": "user", "content": payload}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

- [ ] **Step 3: Commit**

```bash
git add wukong/v2/hannah_memory_scene.ipynb
git commit -m "add create_blueprints function to memory scene notebook"
```

---

### Task 3: Add first-pass run cell and verify output

**Files:**
- Modify: `wukong/v2/hannah_memory_scene.ipynb` (insert after the function cell from Task 2)

- [ ] **Step 1: Read the notebook to get the function cell's ID**

Use the Read tool on `wukong/v2/hannah_memory_scene.ipynb`. Find the cell containing `def create_blueprints`. Call its ID `<fn_cell_id>`.

- [ ] **Step 2: Insert the first-pass run cell after `<fn_cell_id>`**

Use NotebookEdit with `edit_mode: insert`, `cell_type: code`, `cell_id: <fn_cell_id>`:

```python
# First pass — run for one event and inspect before saving
target_event = key_events["events"][0]
all_events = key_events["events"]

blueprints_result = create_blueprints(target_event, all_events)
print(json.dumps(blueprints_result, indent=2))
```

- [ ] **Step 3: Run the cell in the notebook and inspect output**

Execute the cell. Verify:
- Output is valid JSON with a `"blueprints"` key
- Each blueprint has: `id`, `time`, `interlocutor`, `location`, `emotional_state`, `description`
- `id` follows format `{event_id}_{time}_{interlocutor}`
- `time` is one of `before`, `during`, `immediately_after`, `long_after`
- `description` is 3–6 sentences in third-person referring to Hannah by name
- Blueprint count is between 4 and 8

- [ ] **Step 4: Commit**

```bash
git add wukong/v2/hannah_memory_scene.ipynb
git commit -m "add first-pass blueprint run cell to memory scene notebook"
```

---

### Task 4: Add save cells (single event + batch all events)

**Files:**
- Modify: `wukong/v2/hannah_memory_scene.ipynb` (insert after the run cell from Task 3)

- [ ] **Step 1: Read the notebook to get the run cell's ID**

Use the Read tool on `wukong/v2/hannah_memory_scene.ipynb`. Find the cell containing `blueprints_result = create_blueprints`. Call its ID `<run_cell_id>`.

- [ ] **Step 2: Insert the save-one cell after `<run_cell_id>`**

Use NotebookEdit with `edit_mode: insert`, `cell_type: code`, `cell_id: <run_cell_id>`:

```python
# Run this cell only if the output above looks good — saves the first-pass result
import os

BLUEPRINTS_DIR = "../../data/scenes/blueprints"
os.makedirs(BLUEPRINTS_DIR, exist_ok=True)

event_id = target_event["id"]
out_path = os.path.join(BLUEPRINTS_DIR, f"{event_id}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(blueprints_result, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}")
```

- [ ] **Step 3: Read the notebook to get the save-one cell's ID**

Use the Read tool and find the cell just inserted. Call its ID `<save_one_cell_id>`.

- [ ] **Step 4: Insert the batch-all cell after `<save_one_cell_id>`**

Use NotebookEdit with `edit_mode: insert`, `cell_type: code`, `cell_id: <save_one_cell_id>`:

```python
# Batch — process all events and save each to data/scenes/blueprints/{event_id}.json
import os

BLUEPRINTS_DIR = "../../data/scenes/blueprints"
os.makedirs(BLUEPRINTS_DIR, exist_ok=True)

all_events = key_events["events"]
for event in all_events:
    result = create_blueprints(event, all_events)
    out_path = os.path.join(BLUEPRINTS_DIR, f"{event['id']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {event['id']} ({len(result['blueprints'])} blueprints)")
```

- [ ] **Step 5: Commit**

```bash
git add wukong/v2/hannah_memory_scene.ipynb
git commit -m "add blueprint save cells (single event and batch) to memory scene notebook"
```
