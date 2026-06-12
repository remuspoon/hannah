# Scene Blueprint Generation — Design Spec
**Date:** 2026-06-12

## Overview

Given a single key event from Hannah's autobiography, generate a set of scene blueprint objects. Each blueprint describes a specific memory scene — the context, participants, timing, and emotional state — that will serve as the foundation for script generation downstream.

---

## Input

One API call per key event. The model receives:

```json
{
  "target_event": {
    "id": "father_abandonment_age_12",
    "tags": ["family_father"],
    "description": "..."
  },
  "all_events": [ ...full key events list... ]
}
```

- `target_event` — the event for which blueprints are being generated
- `all_events` — the full key events list, provided as continuity context only. The model must not generate blueprints for any event other than `target_event`.

The raw autobiography JSON is not passed — the key events already distil it.

---

## Output

```json
{
  "blueprints": [
    {
      "id": "father_abandonment_age_12_during_father",
      "time": "during",
      "interlocutor": "father",
      "location": "family home, hallway",
      "emotional_state": "confused, pleading",
      "description": "..."
    }
  ]
}
```

---

## Blueprint Object Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | `{event_id}_{time}_{interlocutor}` — unique, snake_case |
| `time` | enum | One of: `before`, `during`, `immediately_after`, `long_after` |
| `interlocutor` | string | The person Hannah is with in the scene — derived from the event |
| `location` | string | Where the scene takes place — inferred to fit the time/interlocutor combination |
| `emotional_state` | string | Hannah's emotional state in this specific scene |
| `description` | string | 3–6 sentence narrative scene-setter (see below) |

---

## Permutation Strategy

### Primary axes: time × interlocutor
The model generates one blueprint per combination of temporal position and interlocutor. Interlocutors are inferred from whoever is meaningfully present in the target event.

- **Time positions:** `before`, `during`, `immediately_after`, `long_after`
- **Interlocutors:** event-derived (e.g., for father abandonment: the father; for bullying: the bully and a teacher)

For an event with one interlocutor: 4 blueprints.  
For an event with two interlocutors: up to 8 blueprints.

### Emotional state variation (single-interlocutor events)
When an event has only one interlocutor, the `long_after` temporal slot should produce two blueprints with contrasting emotional states (e.g., numb vs. angry) to add meaningful variety without inventing characters.

---

## Description Field

The description is a 3–6 sentence narrative written in close third-person, referring to Hannah by name. It extends the key event's description but is recontextualised to fit the specific time, location, interlocutor, and emotional state of this blueprint.

**Allowed:** sensory/emotional texture, plausible setting details consistent with the autobiography  
**Not allowed:** new characters, contradictions of stated facts, changes to event outcomes

The description must carry continuity from prior events — using `all_events` as context, later scenes in Hannah's life should reflect the cumulative weight of what came before.

---

## Continuity Rules

- `all_events` is ordered chronologically. The model uses events prior to `target_event` to inform Hannah's baseline state, and events after to avoid contradicting her future.
- Descriptions must feel like one coherent life, not isolated vignettes.
- The `long_after` blueprints in particular should reflect how Hannah has (or hasn't) processed the event given everything else she has experienced.

---

## Implementation Location

`wukong/v2/hannah_memory_scene.ipynb` — new cells added after the existing key events generation cells.

The function signature will be:
```python
def create_blueprints(target_event: dict, all_events: list) -> dict:
    ...
```

Results stored in a variable for inspection before saving to `data/scenes/blueprints/{event_id}.json`.
