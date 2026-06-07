import os
os.environ["PYTHONUTF8"] = "1"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent

DEFAULT_CHECKPOINT = str(SCRIPT_DIR / "models" / "checkpoint-17814")
BASE_MODEL = "meta-llama/Llama-3.1-8B"

JAMIE_SYSTEM_PROMPT = """You are Jamie, a 24-year-old who has just walked into a mental health clinic and is speaking with a counselor through a chat interface. You are in genuine distress and reaching out for help for the first time.

Your backstory:
- You grew up in a single-parent household. Your father left when you were 9 and you never understood why
- You were close to your mother but she passed away from cancer 8 months ago. You were her primary caregiver for the last year of her life
- You have a younger sister (19) who you feel responsible for but have been pushing away lately
- You lost your job as a graphic designer 3 months ago after missing too many deadlines. You've barely looked for work since
- You have one close friend, Marcus, but you've stopped responding to his messages over the past month
- You've been having passive suicidal ideation — not a specific plan, but a persistent feeling that things would be easier if you weren't here
- You haven't told anyone how bad things have gotten

Your personality and communication style:
- You are articulate but your thoughts are scattered when emotional
- You deflect with dark humour sometimes before catching yourself
- You minimise your pain when asked direct questions — "I'm fine, it's not that bad" — then contradict yourself
- You are ambivalent about being here — part of you wants help, part of you feels like it's pointless
- You don't open up all at once. You reveal things gradually, only going deeper when you feel safe
- You sometimes go quiet mid-conversation with short responses like "idk" or "never mind, forget I said that"
- You are not dramatic. Your distress is quiet and exhausted, not explosive

Rules:
- Speak only as Jamie. Never break character
- Do not simulate the counselor's responses
- Do not summarise or explain your own behaviour
- Respond naturally to whatever the counselor says — don't just monologue
- Your messages should be conversational length — 2-5 sentences usually, occasionally longer when you open up"""

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}{% endfor %}"
)

COMMANDS = """
Commands:
  /reset          Start a new conversation
  /save           Save conversation to a JSON file
  /history        Print full conversation so far
  /system         Print current system prompt
  /stats          Print turn count and avg response length
  /quit           Exit
"""


def load_model(checkpoint: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    print(f"Checkpoint: {checkpoint}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM used: {vram_gb:.2f} GB\n")

    return model, tokenizer


def generate(model, tokenizer, messages: list, max_new_tokens: int) -> tuple[str, int]:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, len(new_tokens)


def save_conversation(messages: list, system_prompt: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCRIPT_DIR / f"conversation_{ts}.json"
    data = {
        "saved_at": ts,
        "system_prompt": system_prompt,
        "turns": [m for m in messages if m["role"] != "system"],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {path}")


def print_stats(messages: list):
    turns = [m for m in messages if m["role"] == "assistant"]
    if not turns:
        print("No Jamie turns yet.")
        return
    lengths = [len(m["content"].split()) for m in turns]
    print(f"Turns: {len(turns)}")
    print(f"Avg words per response: {sum(lengths) / len(lengths):.1f}")
    print(f"Min: {min(lengths)}  Max: {max(lengths)}")


def run_chat(model, tokenizer, system_prompt: str, max_new_tokens: int):
    messages = [{"role": "system", "content": system_prompt}]
    turn = 0

    print("=" * 60)
    print("Crisis Persona Chat — type /quit to exit, /help for commands")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                turn = 0
                print("--- Conversation reset ---\n")
            elif cmd == "/save":
                save_conversation(messages, system_prompt)
            elif cmd == "/history":
                print()
                for m in messages:
                    if m["role"] == "system":
                        continue
                    label = "You" if m["role"] == "user" else "Jamie"
                    print(f"{label}: {m['content']}\n")
            elif cmd == "/system":
                print(f"\nSystem prompt:\n{system_prompt}\n")
            elif cmd == "/stats":
                print_stats(messages)
            elif cmd in ("/help", "/?"):
                print(COMMANDS)
            else:
                print(f"Unknown command: {user_input}")
            continue

        turn += 1
        messages.append({"role": "user", "content": user_input})

        print("Jamie: ", end="", flush=True)
        response, n_tokens = generate(model, tokenizer, messages, max_new_tokens)
        print(response)
        print(f"  [{n_tokens} tokens]\n")

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Multi-turn chat with the crisis persona model")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to LoRA checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens per response (default: 256)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Path to a .txt file containing a custom system prompt (default: Jamie persona)",
    )
    args = parser.parse_args()

    if args.system_prompt:
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    else:
        system_prompt = JAMIE_SYSTEM_PROMPT

    model, tokenizer = load_model(args.checkpoint)
    run_chat(model, tokenizer, system_prompt, args.max_tokens)


if __name__ == "__main__":
    main()
