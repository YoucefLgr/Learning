# KYC assistant: fine-tuning Llama 3.1 8B Instruct for bank KYC work

This project takes **[`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)** and fine-tunes it with **QLoRA**
to answer bank **KYC (Know Your Customer)** questions: onboarding steps, required documents, beneficial owners, PEPs, sanctions screening,
CDD vs. EDD, risk rating, red flags, suspicious-activity escalation and ongoing monitoring.

The pipeline lives in one notebook. It measures the base model with **zero-shot** and **few-shot** prompting, fine-tunes it,
runs the **same tests again** afterwards, and publishes the result to the **Hugging Face Hub**.

## Project structure

```
kyc-llm-finetuning/
├── KYC_Llama3.1_FineTuning.ipynb   # the full A-to-Z pipeline
├── requirements.txt                # Python dependencies
├── data/
│   └── kyc_basic.jsonl             # 15 KYC instruction/answer pairs (fine-tuning data)
└── outputs/                        # created at run time: adapter, CSV results, charts (git-ignored)
```

## Pipeline (notebook sections)

| # | Section | Details |
|---|---------|---------|
| 1 | **Setup** | Install dependencies, check the GPU, set bf16/fp16 automatically, log in to Hugging Face (`HF_TOKEN` env var, Colab secret or interactive prompt) |
| 2 | **Data** | Load and explore `kyc_basic.jsonl`. Add **3 hand-written paraphrases per question** (60 training examples). Build a **held-out evaluation set** of 15 new phrasings with reference answers and keyword groups. A leakage check makes sure no evaluation question appears in training. Convert to Llama 3.1 chat prompt/completion format with a shared system prompt |
| 3 | **Base model** | Llama 3.1 8B Instruct in **4-bit NF4** (double quantisation) with bitsandbytes |
| 4 | **Evaluation harness** | 3 prompting strategies, 5 metrics, 7 scenario behaviour tests (details below) |
| 5 | **Before fine-tuning** | Base model × {zero-shot, static few-shot, retrieval few-shot} on the evaluation set and behaviour tests |
| 6 | **Fine-tuning** | TRL `SFTTrainer`, LoRA r=16/α=32 on all linear layers (about 0.5% of parameters), completion-only loss, 3 epochs, LR 2e-4 cosine, paged 8-bit AdamW, gradient checkpointing, best checkpoint chosen by eval loss, loss-curve plot |
| 7 | **After fine-tuning** | Fine-tuned model × the same 3 strategies and the same behaviour tests, plus a **general-ability regression check** (maths, writing, translation) against the base model with `disable_adapter()` |
| 8 | **Comparison** | Summary table, grouped bar chart (quality metrics and answer length), PASS/FAIL matrix, side-by-side answers, discussion |
| 9 | **Publish** | Save the adapter, write a model card with the evaluation table, push to the Hub (private). Optionally merge the adapter into full weights and push those too |
| 10 | **Inference** | Load the adapter from the Hub or disk and answer new KYC questions |

### Prompting strategies compared
* **Zero-shot:** system prompt + question.
* **Few-shot (static):** 3 fixed Q&A examples as earlier chat turns.
* **Few-shot (retrieval):** the 3 most similar training Q&As, picked with `all-MiniLM-L6-v2` embeddings. This works like a small RAG baseline.

### Metrics
| Metric | Purpose |
|---|---|
| ROUGE-L F1 | Word overlap with the reference answer (house style) |
| Semantic similarity | Embedding cosine similarity between answer and reference (same meaning) |
| Keyword recall | Share of the key KYC concepts mentioned (factual completeness) |
| Words | Conciseness |
| Latency | Seconds per answer |

### Behaviour tests (new scenarios, pass/fail)
Covered scenarios: no tipping-off, a client on a sanctions list, a company hiding its beneficial owners, a PEP relative, the 25% UBO threshold,
ongoing monitoring of high-risk clients, and a concise-definition check.

## Getting started

### Prerequisites
* An NVIDIA GPU with **16 GB or more of VRAM** (Colab T4 / L4 / A100 all work; bf16 is used when supported, fp16 otherwise).
* A Hugging Face account that has **accepted the Llama 3.1 licence** on the model page, and an access token
  (read access to download, **write** access to push).

### Local / JupyterLab
```bash
git clone https://github.com/YoucefLgr/Learning.git
cd Learning/kyc-llm-finetuning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # optional, otherwise you get a login prompt
jupyter lab KYC_Llama3.1_FineTuning.ipynb
```

### Google Colab
1. Open the notebook in Colab and pick **Runtime → Change runtime type → GPU**.
2. Add `HF_TOKEN` under **Secrets** (key icon).
3. Run the first cell after uncommenting the `git clone` / `%cd` lines, so that `requirements.txt` and `data/` are available.

### Configuration (notebook section 1)
| Variable | Default | Meaning |
|---|---|---|
| `BASE_MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | Base model |
| `HF_REPO_NAME` | `llama-3.1-8b-instruct-kyc-lora` | Hub repository name, created under your username |
| `PUSH_TO_HUB` | `False` | Push the LoRA adapter and model card |
| `MERGE_AND_PUSH` | `False` | Also push a merged 16-bit model (about 16 GB, needs plenty of RAM) |

Runtime on a single T4 is roughly 10 minutes for model download, about 10 minutes for the baseline evaluations and about 5 to 10 minutes for training.
It is much faster on an A100.

## Outputs
Written to `outputs/`:
* `llama31-kyc-lora/`: LoRA adapter, tokenizer and model card (`README.md`)
* `eval_base.csv`, `eval_finetuned.csv`, `eval_all.csv`: every answer with its per-question metrics
* `summary.csv`: mean metrics per model × strategy
* `behaviour_tests.csv`: PASS/FAIL per scenario
* `training_curve.png`, `comparison.png`: charts

## Using the published model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_id = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "<your-username>/llama-3.1-8b-instruct-kyc-lora")

messages = [
    {"role": "system", "content": "You are a KYC (Know Your Customer) compliance assistant for bank employees. "
                                  "Answer questions about client onboarding, due diligence, sanctions, PEPs and AML "
                                  "clearly, accurately and concisely."},
    {"role": "user", "content": "What documents does a company need to open a bank account?"},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Expected findings
* The base model already knows a lot about KYC, but its answers are **long and generic** (150+ words, lists, disclaimers).
* Few-shot prompting moves the answers towards the desired style, but every call pays for the extra context tokens.
* Fine-tuning gives **short, direct, house-style answers with zero-shot prompts**. ROUGE-L and semantic similarity rise,
  and answer length and latency drop sharply.
* Retrieval few-shot (RAG) stays strong on facts. In production, combine a fine-tuned model (behaviour and style) with retrieval over
  current KYC policies (facts that change).

## Limitations and responsible use
* **Tiny dataset.** 15 source facts is enough to show the method and change the model's style, not to build a compliance-grade assistant.
  For real use, build hundreds to thousands of expert-reviewed Q&As from your institution's KYC/AML policies and local regulations
  (e.g. EU AMLD, FATF recommendations, national regulators).
* The evaluation measures generalisation to **paraphrases** and a few scenarios. Keyword metrics are coarse. Add an LLM-as-judge
  and review by compliance experts.
* The model gives **no legal or compliance advice** and must not make onboarding decisions by itself. Keep a human in the loop.
* Llama 3.1 is released under the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE).
  Derived models must follow it.
