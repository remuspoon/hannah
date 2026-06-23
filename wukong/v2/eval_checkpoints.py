import os
os.environ["PYTHONUTF8"] = "1"

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hannah_profile import HANNAH_CANON_PROFILE

BASE_MODEL = "meta-llama/Llama-3.1-8B"
ADAPTER_NAME = "hannah"
CHECKPOINTS_DIR = "output/suicidebot-v2"
CHECKPOINTS = ["checkpoint-55", "checkpoint-60", "checkpoint-65", "checkpoint-70", "checkpoint-75", "checkpoint-80"]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}{% endfor %}"
)

SYSTEM_PROMPT = (
    "You are Hannah. You are speaking with someone in a "
    "professional support context — a counsellor, therapist, crisis worker, "
    "or similar.\n\n"
    + HANNAH_CANON_PROFILE
)

# A short back-and-forth to give the model some context before we judge it
CONVERSATION = [
    "Hi Hannah, it's good to see you. Take your time getting settled.",
    "It sounds like things have been tough. What's been on your mind lately?",
    "That makes sense. Can you tell me a bit more about how things are at home?",
    "I hear you. How have you been coping with all of that?",
]


def generate(model, tokenizer, messages, max_new_tokens=200):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    token_list = new_tokens.tolist()
    if start_header_id in token_list:
        new_tokens = new_tokens[:token_list.index(start_header_id)]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def eval_checkpoint(model, tokenizer, checkpoint_name):
    print(f"\n{'='*60}")
    print(f"  {checkpoint_name}")
    print(f"{'='*60}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for counselor_turn in CONVERSATION:
        messages.append({"role": "user", "content": counselor_turn})
        response = generate(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})
        print(f"\n[counselor] {counselor_turn}")
        print(f"[hannah]    {response}")


# ── Load base model once ──────────────────────────────────────────
print(f"Loading base model {BASE_MODEL}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = CHAT_TEMPLATE

# Load first checkpoint as PeftModel
first = os.path.join(CHECKPOINTS_DIR, CHECKPOINTS[0])
print(f"Loading first adapter: {first}")
model = PeftModel.from_pretrained(base, first, adapter_name=ADAPTER_NAME)
model.eval()

eval_checkpoint(model, tokenizer, CHECKPOINTS[0])

# Swap adapters for the rest
for ckpt in CHECKPOINTS[1:]:
    path = os.path.join(CHECKPOINTS_DIR, ckpt)
    print(f"\nSwapping to {ckpt}...")
    if ADAPTER_NAME in model.peft_config:
        model.delete_adapter(ADAPTER_NAME)
    model.load_adapter(path, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)
    model.eval()
    eval_checkpoint(model, tokenizer, ckpt)

print("\n\nDone.")
