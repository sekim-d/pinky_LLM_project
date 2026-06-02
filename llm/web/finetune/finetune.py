"""
qwen2.5:3b 파인튜닝 스크립트 (Unsloth + QLoRA)

설치:
  pip install unsloth datasets trl torch

실행:
  python finetune.py

완료 후:
  ./import_to_ollama.sh
"""

import json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME  = "unsloth/Qwen2.5-3B-Instruct"
OUTPUT_DIR  = "./output"
DATA_PATH   = "./training_data.jsonl"
MAX_SEQ_LEN = 512

# ── 모델 로드 (4bit 양자화로 VRAM 절약) ──────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# ── QLoRA 적용 ────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    lora_alpha          = 16,
    lora_dropout        = 0,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
)

# ── 데이터 로드 ───────────────────────────────────────────────
def load_data(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            rows.append({"text": text})
    return Dataset.from_list(rows)

dataset = load_data(DATA_PATH)
print(f"학습 데이터: {len(dataset)}개")

# ── 학습 ─────────────────────────────────────────────────────
trainer = SFTTrainer(
    model   = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field    = "text",
        max_seq_length        = MAX_SEQ_LEN,
        output_dir            = OUTPUT_DIR,
        num_train_epochs      = 3,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        learning_rate         = 2e-4,
        fp16                  = True,
        logging_steps         = 10,
        save_steps            = 50,
        warmup_ratio          = 0.1,
        lr_scheduler_type     = "cosine",
    ),
)

trainer.train()

# ── GGUF 저장 (Ollama 임포트용) ───────────────────────────────
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method = "q4_k_m",  # 4bit 양자화 (속도/품질 균형)
)

print(f"\n완료! GGUF 파일: {OUTPUT_DIR}/")
print("다음 단계: ./import_to_ollama.sh 실행")
