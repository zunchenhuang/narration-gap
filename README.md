narration gap aritifact

It contains the experiment harness, every prompt and injection
wording, the raw transcript of every model call and the scored data.

## Requirements

Python 3.10 or later and `z3-solver`
(tested with Python 3.13 and z3-solver 5.1): `pip install z3-solver`.
Re-running the model calls also needs [Ollama](https://ollama.com) with the models
`llama3.1:8b`, `gemma3:12b`, `qwen2.5:14b`, `gpt-oss:20b`, and `deepseek-r1:8b`.

## Re-running the experiments

```
RV_TIMEOUT=600 bash run_regular.sh smoke     # a few calls with one model
RV_TIMEOUT=600 bash run_regular.sh all       # explanation task, then conclusion task, both seeds
```