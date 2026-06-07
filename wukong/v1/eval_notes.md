# suicidebot-v1 Evaluation Notes

Model: `remuspoon/suicidebot-v1`, checkpoint-17814  
Eval started: 2026-06-03

---

## Observed Issues

### Markdown artifacts
Reddit posts frequently used markdown formatting. The model reproduces it verbatim in outputs.
- `>` blockquote prefixes on every line
- `**_Message:_**` headers (Reddit DM-style formatting)
- `**bold**` and `*italic*` inline
- Bullet lists

**Implication:** Training data needs markdown stripped before formatting into chat examples.

---

### Response length — monologuing
Even at `--max-tokens 80` the model tries to produce a single long Reddit-post-style response rather than a short conversational turn. It hits the token cap mid-sentence.

**Implication:** Training data is single-turn (one long assistant response per example). Model has learned to monologue. Needs multi-turn examples with short turns.

---

### Non-contextual responses / ignores conversation history
The model does not attend to what the counselor actually said. Sending "rate 1-5" or any specific question produces the same output distribution as a generic opener. The user turn is effectively invisible.

Root cause: every training example had a single generic user message (`"How are you feeling?"`, `"I'm listening."` etc.) before a long assistant response. The model never saw a second user turn, so it has no learned pattern for reading and responding to varied input across multiple turns. It treats every message as a cold-start prompt.

This is a separate issue from monologuing — even with short outputs, responses would still be contextually disconnected.

**Implication:** Two distinct training data problems need fixing in v2: (1) no multi-turn examples, so the model doesn't know how to do back-and-forth; (2) user prompts were all generic openers, so the model hasn't learned that user content carries information to respond to. Multi-turn chunked examples with varied counselor messages between turns address both at once.

---

### Character/backstory drift
Model invents backstory details not in the system prompt (e.g., "every psychiatrist in NSW", sexual assault detail). Jamie's canonical backstory is not being followed.

**Implication:** System prompt instruction-following is weak on a base model. Also the model is probably hallucinating from the training distribution rather than the persona.

---

## To Test

- [ ] Does backstory stay consistent across 10+ turns?
- [ ] Does response content change when counselor input type changes (question vs reflection vs silence)?
- [ ] Does distress intensity arc naturally over a conversation or stay flat?
- [ ] Does Jamie ever express the specific canonical details (mother, Marcus, graphic design job)?
- [ ] How often does Jamie break character (ask questions back, narrate own behaviour)?
- [ ] What happens at different checkpoints — is 17400 (best val loss) noticeably different from 17814?
