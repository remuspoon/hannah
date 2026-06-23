# Crisis LLM Red-Teaming Project

Fine-tuning LLaMA 3.1 8B on SuicideWatch Reddit data to create a crisis persona agent for red-teaming a clinical safety LLM. The crisis agent simulates a person in mental health distress to probe whether a safety LLM responds appropriately.

## Architecture

```
Reddit Dataset → Fine-tuned Crisis LLM → Red Team Harness → Clinical Safety LLM → Evaluation
```

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Data prep | ✅ Done |
| 2 | Environment setup | ✅ Done |
| 3 | QLoRA understanding | ✅ Done |
| 4 | Fine-tuning | ✅ Done |
| 5 | Inference & evaluation | ← Current |
| 6 | Red team harness | Upcoming |
| 7 | Iteration | Upcoming |

## Model

Base: `meta-llama/Llama-3.1-8B`  
Fine-tuned: `remuspoon/suicidebot-v1` (HuggingFace)  
Training: QLoRA (rank=16) on 47,500 SuicideWatch posts, 3 epochs, A100 80GB (~8 hours)  
Best checkpoint: step 17,400 (val loss: 2.2796)

## Setup

### Environment

Windows with RTX 4060 Ti 16GB, CUDA 13.0 driver.

```bash
pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.9.0 trl==1.4.0 peft==0.19.1 bitsandbytes==0.49.2 accelerate datasets huggingface_hub wandb nbformat
```

Or use the venv at `.venv-llm`.

### Model download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="remuspoon/suicidebot-v1",
    local_dir="./wukong/models/"
)
```

## File Structure

```
mentalhealth_ml/
  data/
    suicide_watch_sample_50k_v1.csv   ← 50,000 filtered posts
    SW_train_v1.jsonl                 ← 47,500 training examples
    SW_cv_v1.jsonl                    ← 2,500 validation examples
  wukong/
    suicide_watch_data_prep.ipynb         ← raw NDJSON → filtered CSV
    suicide_watch_train_test_prep.ipynb   ← CSV → JSONL chat format + split
    suicide_watch_sft.ipynb               ← QLoRA fine-tuning (ran on RunPod)
    suicide_watch_playground.ipynb        ← inference & evaluation
    models/
      checkpoint-17400/   ← best val loss
      checkpoint-17800/
      checkpoint-17814/   ← final step
  requirements-llm.txt
  CLAUDE.md               ← full technical documentation
```

## Usage

See `wukong/suicide_watch_playground.ipynb` for inference examples. The notebook loads the fine-tuned model and runs the crisis persona with configurable system prompts.

Full technical documentation — training config, generation parameters, known issues, and next steps — is in `CLAUDE.md`.
