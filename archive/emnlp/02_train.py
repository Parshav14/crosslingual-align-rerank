import os
import gc
import csv
import json
import time
import random
import warnings
import traceback
import numpy as np
import pandas as pd
import nltk
import torch
import evaluate
import torch.nn.functional as F

from math import ceil
from datetime import datetime
from tqdm.auto import tqdm

from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)

warnings.filterwarnings("ignore")

# =====================================================
# 1. H100 MAXIMUM STABILITY
# =====================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("🚀 EMNLP FACTUAL MT5 TRAINING")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 80)

# =====================================================
# 2. PATHS
# =====================================================
BASE_DIR = "/workspace/jasmeet/entailment_contrastive"

FACTUAL_PATH = f"{BASE_DIR}/dataset_splits/factual_train"

HALLUCINATED_PATH = f"{BASE_DIR}/dataset_splits/hallucinated_train"

VAL_PATH = f"{BASE_DIR}/dataset_splits/val_mixed"

OUTPUT_DIR = f"{BASE_DIR}/emnlp_mt5_factual_7lang"

FINAL_MODEL_DIR = f"{BASE_DIR}/emnlp_final_mt5_factual_7lang"

# =====================================================
# LOGGING DIRECTORIES
# =====================================================
LOG_DIR = f"{BASE_DIR}/training_logs"

CSV_DIR = f"{LOG_DIR}/csv_logs"

JSON_DIR = f"{LOG_DIR}/json_logs"

REPORT_DIR = f"{LOG_DIR}/reports"

CHECKPOINT_LOG_DIR = f"{LOG_DIR}/checkpoint_logs"

for d in [
    LOG_DIR,
    CSV_DIR,
    JSON_DIR,
    REPORT_DIR,
    CHECKPOINT_LOG_DIR
]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = f"{CSV_DIR}/training_metrics.csv"

EVAL_CSV = f"{CSV_DIR}/eval_metrics.csv"

GPU_CSV = f"{CSV_DIR}/gpu_memory.csv"

FINAL_JSON = f"{JSON_DIR}/final_metrics.json"

TRAINING_REPORT = f"{REPORT_DIR}/training_report.txt"

# =====================================================
# 3. LOAD MODEL
# =====================================================
MODEL_NAME = "google/mt5-large"

print(f"\n📥 Loading model:")
print(MODEL_NAME)

config = AutoConfig.from_pretrained(
    MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    config=config
).to(device)

model.gradient_checkpointing_enable()

model.config.use_cache = False

# =====================================================
# 4. LOAD DATASETS
# =====================================================
print("\n📚 Loading datasets...")

train_dataset = load_from_disk(
    FACTUAL_PATH
)

hall_dataset = load_from_disk(
    HALLUCINATED_PATH
)

val_dataset = load_from_disk(
    VAL_PATH
)

if len(val_dataset) > 2000:

    val_dataset = val_dataset.shuffle(
        seed=42
    ).select(
        range(2000)
    )

print(f"✅ Factual Train: {len(train_dataset)}")
print(f"❌ Hallucinated Train: {len(hall_dataset)}")
print(f"🟡 Validation: {len(val_dataset)}")

# =====================================================
# 5. TOKENIZATION
# =====================================================
MAX_INPUT = 640

MAX_TARGET = 128

def preprocess(batch):

    try:

        model_inputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_INPUT
        )

        labels = tokenizer(
            text_target=batch["summary"],
            truncation=True,
            max_length=MAX_TARGET
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    except Exception:

        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

print("\n🔄 Tokenizing datasets...")

tokenized_train = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=4
)

tokenized_val = val_dataset.map(
    preprocess,
    batched=True,
    remove_columns=val_dataset.column_names,
    num_proc=4
)

gc.collect()
torch.cuda.empty_cache()

# =====================================================
# 6. METRICS
# =====================================================
nltk.download("punkt", quiet=True)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.asarray(
        predictions
    ).astype(np.int64)

    labels = np.asarray(
        labels
    ).astype(np.int64)

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id
    )

    predictions = np.clip(
        predictions,
        0,
        tokenizer.vocab_size - 1
    )

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    decoded_preds = [
        "\n".join(
            nltk.sent_tokenize(p.strip())
        )
        for p in decoded_preds
    ]

    decoded_labels = [
        "\n".join(
            nltk.sent_tokenize(l.strip())
        )
        for l in decoded_labels
    ]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    result = {
        k: round(v * 100, 4)
        for k, v in result.items()
    }

    return result

# =====================================================
# 7. CSV LOGGER
# =====================================================
class CSVLoggerCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return

        try:

            logs["step"] = state.global_step

            logs["epoch"] = (
                round(state.epoch, 4)
                if state.epoch is not None
                else 0
            )

            logs["gpu_memory_gb"] = round(
                torch.cuda.memory_allocated() / 1e9,
                4
            )

            # =========================================
            # TRAIN CSV
            # =========================================
            train_exists = os.path.isfile(
                TRAIN_CSV
            )

            with open(
                TRAIN_CSV,
                "a",
                newline=""
            ) as csvfile:

                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=logs.keys()
                )

                if not train_exists:
                    writer.writeheader()

                writer.writerow(logs)

            # =========================================
            # GPU CSV
            # =========================================
            gpu_exists = os.path.isfile(
                GPU_CSV
            )

            gpu_data = {
                "step": state.global_step,
                "gpu_memory_gb": logs["gpu_memory_gb"],
                "time": time.time()
            }

            with open(
                GPU_CSV,
                "a",
                newline=""
            ) as gpu_csv:

                writer = csv.DictWriter(
                    gpu_csv,
                    fieldnames=gpu_data.keys()
                )

                if not gpu_exists:
                    writer.writeheader()

                writer.writerow(gpu_data)

        except Exception as e:

            print(f"⚠️ Logging Error: {e}")

# =====================================================
# 8. EMNLP FACTUAL TRAINER
# =====================================================
class EMNLPFactualTrainer(Seq2SeqTrainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(
            **inputs,
            labels=labels,
            output_hidden_states=True
        )

        ce_loss = outputs.loss

        decoder_hidden = outputs.decoder_hidden_states[-1]

        pooled = decoder_hidden.mean(dim=1)

        shuffled = pooled[
            torch.randperm(
                pooled.size(0)
            )
        ]

        pos_sim = F.cosine_similarity(
            pooled,
            pooled.detach(),
            dim=-1
        )

        neg_sim = F.cosine_similarity(
            pooled,
            shuffled.detach(),
            dim=-1
        )

        margin = 0.30

        contrastive_loss = torch.clamp(
            margin - pos_sim + neg_sim,
            min=0
        ).mean()

        consistency_loss = (
            1 - pos_sim.mean()
        )

        ALPHA = 0.10

        BETA = 0.05

        total_loss = (
            ce_loss +
            (ALPHA * contrastive_loss) +
            (BETA * consistency_loss)
        )

        if torch.isnan(total_loss):

            print("⚠️ NaN detected.")

            total_loss = ce_loss

        self.log({

            "ce_loss":
                ce_loss.detach().item(),

            "contrastive_loss":
                contrastive_loss.detach().item(),

            "consistency_loss":
                consistency_loss.detach().item(),

            "total_loss":
                total_loss.detach().item()
        })

        return (
            (total_loss, outputs)
            if return_outputs
            else total_loss
        )

# =====================================================
# 9. HYPERPARAMETERS
# =====================================================
epochs = 2

batch_size = 4

grad_accum = 8

total_steps = ceil(
    len(tokenized_train) /
    (batch_size * grad_accum)
) * epochs

warmup_steps = int(
    0.10 * total_steps
)

print("\n📊 TRAINING CONFIG")

print(f"Epochs: {epochs}")

print(f"Batch Size: {batch_size}")

print(f"Gradient Accumulation: {grad_accum}")

print(f"Warmup Steps: {warmup_steps}")

# =====================================================
# 10. TRAINING ARGUMENTS
# =====================================================
training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_DIR,

    overwrite_output_dir=False,

    logging_steps=25,

    save_strategy="steps",

    save_steps=500,

    save_total_limit=3,

    save_safetensors=False,

    save_only_model=True,

    evaluation_strategy="steps",

    eval_steps=500,

    load_best_model_at_end=True,

    metric_for_best_model="rougeL",

    greater_is_better=True,

    bf16=True,

    tf32=True,

    gradient_checkpointing=True,

    per_device_train_batch_size=batch_size,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=grad_accum,

    learning_rate=3e-5,

    num_train_epochs=epochs,

    warmup_steps=warmup_steps,

    weight_decay=0.01,

    lr_scheduler_type="cosine",

    predict_with_generate=True,

    generation_max_length=128,

    generation_num_beams=4,

    max_grad_norm=1.0,

    dataloader_num_workers=4,

    dataloader_pin_memory=True,

    logging_first_step=True,

    remove_unused_columns=True,

    report_to=[],

    fp16=False
)

# =====================================================
# 11. TRAINER
# =====================================================
trainer = EMNLPFactualTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_val,

    tokenizer=tokenizer,

    data_collator=DataCollatorForSeq2Seq(

        tokenizer,

        model=model,

        label_pad_token_id=-100
    ),

    compute_metrics=compute_metrics,

    callbacks=[

        EarlyStoppingCallback(
            early_stopping_patience=3
        ),

        CSVLoggerCallback()
    ]
)

# =====================================================
# 12. MEMORY CLEANUP
# =====================================================
gc.collect()

torch.cuda.empty_cache()

# =====================================================
# 13. TRAIN
# =====================================================
print("\n🚀 STARTING TRAINING")

training_start = time.time()

try:

    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(
        OUTPUT_DIR
    )

    if last_checkpoint is not None:

        print(f"🔄 Resuming from checkpoint:")
        print(last_checkpoint)

        trainer.train(
            resume_from_checkpoint=last_checkpoint
        )

    else:

        print("🚀 Starting fresh training...")

        trainer.train()

except Exception as e:

    print(f"\n❌ Training crashed:")

    print(traceback.format_exc())

    torch.cuda.empty_cache()

    gc.collect()

    raise e

training_end = time.time()

# =====================================================
# 14. SAVE FINAL MODEL
# =====================================================
print("\n💾 Saving final model...")

for name, param in model.named_parameters():

    if not param.data.is_contiguous():

        param.data = param.data.contiguous()

trainer.save_model(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)

trainer.state.save_to_json(

    os.path.join(
        FINAL_MODEL_DIR,
        "trainer_state.json"
    )
)

# =====================================================
# 15. SAVE FINAL REPORT
# =====================================================
final_report = {

    "model": MODEL_NAME,

    "epochs": epochs,

    "batch_size": batch_size,

    "grad_accum": grad_accum,

    "learning_rate": 3e-5,

    "train_size": len(train_dataset),

    "val_size": len(val_dataset),

    "training_hours":
        round(
            (training_end - training_start) / 3600,
            4
        ),

    "final_checkpoint":
        FINAL_MODEL_DIR
}

with open(
    FINAL_JSON,
    "w"
) as f:

    json.dump(
        final_report,
        f,
        indent=4
    )

with open(
    TRAINING_REPORT,
    "w"
) as f:

    for k, v in final_report.items():

        f.write(f"{k}: {v}\n")

# =====================================================
# 16. FINAL SUMMARY
# =====================================================
print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE")
print("=" * 80)

print(f"\n✅ Final Model:")
print(FINAL_MODEL_DIR)

print(f"\n📈 Training CSV:")
print(TRAIN_CSV)

print(f"\n📈 GPU CSV:")
print(GPU_CSV)

print(f"\n📄 Final JSON:")
print(FINAL_JSON)

print(f"\n📄 Training Report:")
print(TRAINING_REPORT)

print("\n🚀 READY FOR FILE 3")