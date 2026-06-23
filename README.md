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
Training: QLoRA (rank=16) on SuicideWatch posts, 3 epochs

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
