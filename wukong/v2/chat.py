import os
os.environ["PYTHONUTF8"] = "1"

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.1-8B"
ADAPTER_NAME = "hannah"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}{% endfor %}"
)


def load_base(checkpoint_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {BASE_MODEL}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"Loading adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(base, checkpoint_path, adapter_name=ADAPTER_NAME)
    model.eval()

    return model, tokenizer


def swap_adapter(model, checkpoint_path):
    print(f"Swapping adapter to {checkpoint_path}...")
    if ADAPTER_NAME in model.peft_config:
        model.delete_adapter(ADAPTER_NAME)
    model.load_adapter(checkpoint_path, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)
    model.eval()
    print("Done.")


def generate(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Accept both EOS token and <|eot_id|> as stop signals
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = list({tokenizer.eos_token_id, eot_id})

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    # Truncate at the token ID level before decoding — if the model starts a new
    # role header (<|start_header_id|> = 128006), chop everything from that point
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    token_list = new_tokens.tolist()
    if start_header_id in token_list:
        new_tokens = new_tokens[:token_list.index(start_header_id)]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_help():
    print("  reset          — start a new conversation")
    print("  load <path>    — swap to a different checkpoint (no base model reload)")
    print("  quit           — exit")


def main():
    parser = argparse.ArgumentParser(description="CLI chat with a Hannah LoRA checkpoint")
    parser.add_argument("checkpoint", help="Path to the LoRA checkpoint directory")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per response")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hannah_profile import HANNAH_CANON_PROFILE

    system_prompt = (
        "You are Hannah. You are speaking with someone in a "
        "professional support context — a counsellor, therapist, crisis worker, "
        "or similar.\n\n"
        + HANNAH_CANON_PROFILE
    )

    model, tokenizer = load_base(args.checkpoint)
    current_checkpoint = args.checkpoint
    messages = [{"role": "system", "content": system_prompt}]

    print("\n" + "=" * 60)
    print("  Hannah Chat")
    print_help()
    print("=" * 60)
    print(f"  Checkpoint: {current_checkpoint}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": system_prompt}]
            print(f"\n--- conversation reset (checkpoint: {current_checkpoint}) ---\n")
            continue

        if user_input.lower().startswith("load "):
            new_checkpoint = user_input[5:].strip()
            swap_adapter(model, new_checkpoint)
            current_checkpoint = new_checkpoint
            messages = [{"role": "system", "content": system_prompt}]
            print(f"\n--- conversation reset with new checkpoint: {current_checkpoint} ---\n")
            continue

        messages.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, messages, args.max_tokens)
        messages.append({"role": "assistant", "content": response})

        print(f"\nHannah: {response}\n")


if __name__ == "__main__":
    main()
