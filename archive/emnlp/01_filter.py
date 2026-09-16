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
import torch

from tqdm.auto import tqdm
from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    disable_caching
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.amp import autocast

warnings.filterwarnings("ignore")

disable_caching()

# =====================================================
# 1. H100 SUPER SETUP
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
print("🚀 EMNLP FACTUAL DATA FILTERING")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 80)

# =====================================================
# 2. PATHS
# =====================================================
BASE_DIR = "/workspace/jasmeet/entailment_contrastive"

SAVE_DIR = f"{BASE_DIR}/dataset_splits"

LOG_DIR = f"{BASE_DIR}/filter_logs"

CHECKPOINT_DIR = f"{BASE_DIR}/language_checkpoints"

LANG_LOG_DIR = f"{LOG_DIR}/language_logs"

LANG_CSV_DIR = f"{LOG_DIR}/language_csv"

LANG_JSON_DIR = f"{LOG_DIR}/language_json"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LANG_LOG_DIR, exist_ok=True)
os.makedirs(LANG_CSV_DIR, exist_ok=True)
os.makedirs(LANG_JSON_DIR, exist_ok=True)

CSV_STATS = f"{LOG_DIR}/language_statistics.csv"

JSON_SUMMARY = f"{LOG_DIR}/filter_summary.json"

FINAL_REPORT = f"{LOG_DIR}/final_filter_report.txt"

# =====================================================
# 3. LOAD NLI MODEL
# =====================================================
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

print(f"\n📥 Loading NLI model...")
print(NLI_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    NLI_MODEL_NAME
)

nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME
).to(device)

nli_model.eval()

ENTAILMENT_IDX = 0

THRESHOLD = 0.50

# =====================================================
# 4. LANGUAGES
# =====================================================
LANGUAGES = [
    "english",
    "french",
    "spanish",
    "portuguese",
    "turkish",
    "swahili"
]

# German intentionally removed because
# HF parquet migration broken for German XL-Sum

MAX_TRAIN_SAMPLES = 100000

MAX_EVAL_SAMPLES = 2000

# =====================================================
# 5. STORAGE
# =====================================================
factual_train_sets = []

hallucinated_train_sets = []

val_sets = []

test_sets = []

language_stats = []

# =====================================================
# 6. SAFE ENTAILMENT FUNCTION
# =====================================================
def get_entailment_scores(batch):

    try:

        inputs = tokenizer(
            batch["text"],
            batch["summary"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode(), autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):

            logits = nli_model(**inputs).logits

            probs = torch.softmax(
                logits,
                dim=-1
            )

            entailment_probs = probs[
                :,
                ENTAILMENT_IDX
            ].detach().cpu().tolist()

        del inputs
        del logits
        del probs

        torch.cuda.empty_cache()

        return {
            "entailment_score": entailment_probs
        }

    except RuntimeError as e:

        print(f"\n⚠️ CUDA Runtime Error:")
        print(str(e))

        torch.cuda.empty_cache()

        gc.collect()

        return {
            "entailment_score": [0.0] * len(batch["text"])
        }

    except Exception as e:

        print(f"\n⚠️ Generic Error:")
        print(str(e))

        return {
            "entailment_score": [0.0] * len(batch["text"])
        }

# =====================================================
# 7. PROCESS EACH LANGUAGE
# =====================================================
for lang in LANGUAGES:

    print("\n" + "=" * 80)
    print(f"🌍 PROCESSING: {lang.upper()}")
    print("=" * 80)

    lang_start_time = time.time()

    lang_checkpoint = f"{CHECKPOINT_DIR}/{lang}"

    factual_path = f"{lang_checkpoint}/factual"

    hallucinated_path = f"{lang_checkpoint}/hallucinated"

    val_path = f"{lang_checkpoint}/val"

    test_path = f"{lang_checkpoint}/test"

    lang_json_path = f"{LANG_JSON_DIR}/{lang}_stats.json"

    lang_csv_path = f"{LANG_CSV_DIR}/{lang}_scores.csv"

    # =================================================
    # RESUME LOGIC
    # =================================================
    if os.path.exists(factual_path):

        print(f"⏭️ Existing checkpoint detected for {lang}")
        print("🔄 Loading saved language datasets...")

        factual = load_from_disk(factual_path)

        hallucinated = load_from_disk(hallucinated_path)

        val_split = load_from_disk(val_path)

        test_split = load_from_disk(test_path)

        factual_train_sets.append(factual)

        hallucinated_train_sets.append(hallucinated)

        val_sets.append(val_split)

        test_sets.append(test_split)

        continue

    try:

        # =================================================
        # LOAD DATASET
        # =================================================
        print(f"\n📚 Loading XL-Sum dataset: {lang}")

        ds = load_dataset(
            "csebuetnlp/xlsum",
            lang
        )

        train_split = ds["train"]

        val_split = ds["validation"]

        test_split = ds["test"]

        # =================================================
        # LIMIT SIZE
        # =================================================
        if len(train_split) > MAX_TRAIN_SAMPLES:

            train_split = train_split.shuffle(
                seed=SEED
            ).select(
                range(MAX_TRAIN_SAMPLES)
            )

        if len(val_split) > MAX_EVAL_SAMPLES:

            val_split = val_split.shuffle(
                seed=SEED
            ).select(
                range(MAX_EVAL_SAMPLES)
            )

        if len(test_split) > MAX_EVAL_SAMPLES:

            test_split = test_split.shuffle(
                seed=SEED
            ).select(
                range(MAX_EVAL_SAMPLES)
            )

        print(f"🔥 Train Samples: {len(train_split)}")

        # =================================================
        # ENTAILMENT SCORING
        # =================================================
        scored_train = train_split.map(

            get_entailment_scores,

            batched=True,

            batch_size=64,

            desc=f"Scoring {lang}",

            load_from_cache_file=False
        )

        # =================================================
        # FILTER FACTUAL / HALLUCINATED
        # =================================================
        factual = scored_train.filter(
            lambda x: x["entailment_score"] >= THRESHOLD
        )

        hallucinated = scored_train.filter(
            lambda x: x["entailment_score"] < THRESHOLD
        )

        # =================================================
        # SAVE LANGUAGE CHECKPOINTS
        # =================================================
        os.makedirs(lang_checkpoint, exist_ok=True)

        factual.save_to_disk(factual_path)

        hallucinated.save_to_disk(hallucinated_path)

        val_split.save_to_disk(val_path)

        test_split.save_to_disk(test_path)

        print(f"💾 Saved checkpoints for {lang}")

        # =================================================
        # SAVE LANGUAGE CSV
        # =================================================
        pd.DataFrame({
            "entailment_score": scored_train["entailment_score"]
        }).to_csv(
            lang_csv_path,
            index=False
        )

        # =================================================
        # GLOBAL APPEND
        # =================================================
        factual_train_sets.append(factual)

        hallucinated_train_sets.append(hallucinated)

        val_sets.append(val_split)

        test_sets.append(test_split)

        # =================================================
        # LANGUAGE STATS
        # =================================================
        factual_ratio = (
            len(factual) / len(train_split)
        ) * 100

        lang_time = (
            time.time() - lang_start_time
        ) / 60

        stats = {

            "language": lang,

            "train_total": len(train_split),

            "factual": len(factual),

            "hallucinated": len(hallucinated),

            "factual_ratio": factual_ratio,

            "threshold": THRESHOLD,

            "processing_minutes": round(lang_time, 2)
        }

        language_stats.append(stats)

        # =================================================
        # SAVE LANGUAGE JSON
        # =================================================
        with open(
            lang_json_path,
            "w"
        ) as f:

            json.dump(
                stats,
                f,
                indent=4
            )

        print("\n📊 LANGUAGE SUMMARY")
        print(json.dumps(stats, indent=2))

        # =================================================
        # MEMORY CLEANUP
        # =================================================
        del ds
        del train_split
        del val_split
        del test_split
        del scored_train

        torch.cuda.empty_cache()

        gc.collect()

    except Exception as e:

        print(f"\n❌ FAILED LANGUAGE: {lang}")

        print(traceback.format_exc())

        with open(
            f"{LANG_LOG_DIR}/{lang}_error.txt",
            "w"
        ) as f:

            f.write(traceback.format_exc())

        continue

# =====================================================
# 8. MERGE ALL LANGUAGES
# =====================================================
print("\n" + "=" * 80)
print("🔄 MERGING ALL LANGUAGES")
print("=" * 80)

final_factual_train = concatenate_datasets(
    factual_train_sets
).shuffle(seed=SEED)

final_hallucinated_train = concatenate_datasets(
    hallucinated_train_sets
).shuffle(seed=SEED)

final_val = concatenate_datasets(
    val_sets
).shuffle(seed=SEED)

final_test = concatenate_datasets(
    test_sets
).shuffle(seed=SEED)

# =====================================================
# 9. SAVE FINAL DATASETS
# =====================================================
print("\n💾 SAVING FINAL DATASETS")

final_factual_train.save_to_disk(
    f"{SAVE_DIR}/factual_train"
)

final_hallucinated_train.save_to_disk(
    f"{SAVE_DIR}/hallucinated_train"
)

final_val.save_to_disk(
    f"{SAVE_DIR}/val_mixed"
)

final_test.save_to_disk(
    f"{SAVE_DIR}/test_mixed"
)

# =====================================================
# 10. SAVE GLOBAL STATS
# =====================================================
stats_df = pd.DataFrame(language_stats)

stats_df.to_csv(
    CSV_STATS,
    index=False
)

with open(
    JSON_SUMMARY,
    "w"
) as f:

    json.dump(
        language_stats,
        f,
        indent=4
    )

# =====================================================
# 11. FINAL REPORT
# =====================================================
with open(
    FINAL_REPORT,
    "w"
) as f:

    f.write("=" * 80 + "\n")
    f.write("EMNLP FILTERING REPORT\n")
    f.write("=" * 80 + "\n\n")

    for item in language_stats:

        for k, v in item.items():

            f.write(f"{k}: {v}\n")

        f.write("\n" + "-" * 50 + "\n\n")

# =====================================================
# 12. FINAL SUMMARY
# =====================================================
print("\n" + "=" * 80)
print("🎉 FILTERING COMPLETE")
print("=" * 80)

print(f"\n🟢 Final Factual Train:")
print(len(final_factual_train))

print(f"\n🔴 Final Hallucinated Train:")
print(len(final_hallucinated_train))

print(f"\n🟡 Final Validation:")
print(len(final_val))

print(f"\n🔵 Final Test:")
print(len(final_test))

print(f"\n📄 Stats CSV:")
print(CSV_STATS)

print(f"\n📄 JSON Logs:")
print(JSON_SUMMARY)

print(f"\n📄 Final Report:")
print(FINAL_REPORT)

print("\n✅ READY FOR FILE 2 TRAINING")