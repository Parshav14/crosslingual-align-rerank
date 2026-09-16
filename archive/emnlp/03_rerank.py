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
import evaluate
import nltk

from tqdm.auto import tqdm

from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)

from torch.amp import autocast

from sentence_transformers import (
    SentenceTransformer,
    util
)

from comet import (
    download_model,
    load_from_checkpoint
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# =====================================================
# 1. H100 STABILITY
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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 90)
print("🚀 EMNLP FINAL MULTI-OBJECTIVE RERANKING")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 90)

# =====================================================
# 2. PATHS
# =====================================================
BASE_DIR = "/workspace/jasmeet/entailment_contrastive"

SUMM_MODEL_PATH = f"{BASE_DIR}/emnlp_final_mt5_factual_7lang"

TEST_PATH = f"{BASE_DIR}/dataset_splits/test_mixed"

RESULTS_DIR = f"{BASE_DIR}/final_results"

CSV_DIR = f"{RESULTS_DIR}/csv_logs"

JSON_DIR = f"{RESULTS_DIR}/json_logs"

REPORT_DIR = f"{RESULTS_DIR}/reports"

PAIRWISE_DIR = f"{RESULTS_DIR}/language_pair_metrics"

for d in [
    RESULTS_DIR,
    CSV_DIR,
    JSON_DIR,
    REPORT_DIR,
    PAIRWISE_DIR
]:
    os.makedirs(d, exist_ok=True)

# =====================================================
# OUTPUT FILES
# =====================================================
QUALITATIVE_CSV = f"{CSV_DIR}/qualitative_examples.csv"

ALL_PREDICTIONS_CSV = f"{CSV_DIR}/all_predictions.csv"

CANDIDATE_CSV = f"{CSV_DIR}/candidate_scores.csv"

GPU_LOG_CSV = f"{CSV_DIR}/gpu_memory.csv"

PAIRWISE_METRICS_CSV = f"{CSV_DIR}/pairwise_language_metrics.csv"

FINAL_METRICS_JSON = f"{JSON_DIR}/final_metrics.json"

CANDIDATE_JSON = f"{JSON_DIR}/candidate_logs.json"

FINAL_REPORT = f"{REPORT_DIR}/evaluation_report.txt"

# =====================================================
# 3. LOAD SUMMARIZATION MODEL
# =====================================================
print("\n📥 Loading trained summarization model...")

summ_tokenizer = AutoTokenizer.from_pretrained(
    SUMM_MODEL_PATH
)

summ_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMM_MODEL_PATH,
    torch_dtype=torch.bfloat16
).to(device)

summ_model.eval()

torch.cuda.empty_cache()
gc.collect()

# =====================================================
# 4. LOAD NLI MODEL
# =====================================================
print("\n🔍 Loading multilingual NLI reranker...")

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

nli_tokenizer = AutoTokenizer.from_pretrained(
    NLI_MODEL_NAME
)

nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME,
    torch_dtype=torch.bfloat16
).to(device)

nli_model.eval()

ENTAILMENT_IDX = 0

# =====================================================
# 5. LOAD SEMANTIC MODEL
# =====================================================
print("\n🧠 Loading multilingual semantic model...")

SIM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

sim_model = SentenceTransformer(
    SIM_MODEL_NAME,
    device=device
)

# =====================================================
# 6. LOAD COMET
# =====================================================
print("\n🌍 Loading COMET metric model...")

comet_model_path = download_model(
    "Unbabel/wmt22-comet-da"
)

comet_model = load_from_checkpoint(
    comet_model_path
)

# =====================================================
# 7. LOAD TEST DATA
# =====================================================
print("\n📚 Loading test dataset...")

test_dataset = load_from_disk(TEST_PATH)

MAX_TEST_SAMPLES = 1000

if len(test_dataset) > MAX_TEST_SAMPLES:

    test_dataset = test_dataset.shuffle(
        seed=42
    ).select(
        range(MAX_TEST_SAMPLES)
    )

print(f"✅ Loaded {len(test_dataset)} test samples")

# =====================================================
# 8. METRICS
# =====================================================
nltk.download("punkt", quiet=True)

rouge = evaluate.load("rouge")

bertscore = evaluate.load("bertscore")

# =====================================================
# 9. STORAGE
# =====================================================
best_summaries = []

ground_truths = []

source_texts = []

candidate_logs = []

gpu_logs = []

pairwise_rows = []

all_entailment_scores = []

all_semantic_scores = []

all_comet_scores = []

# =====================================================
# 10. GENERATION + RERANKING
# =====================================================
print("\n🚀 Starting generation + reranking...")

batch_size = 4

progress_bar = tqdm(

    range(
        0,
        len(test_dataset),
        batch_size
    ),

    desc="Evaluating"
)

start_time = time.time()

for i in progress_bar:

    try:

        batch = test_dataset[
            i:i + batch_size
        ]

        texts = batch["text"]

        refs = batch["summary"]

        source_texts.extend(texts)

        ground_truths.extend(refs)

        # =================================================
        # TOKENIZATION
        # =================================================
        inputs = summ_tokenizer(

            texts,

            return_tensors="pt",

            padding=True,

            truncation=True,

            max_length=640

        ).to(device)

        # =================================================
        # GENERATION
        # =================================================
        with torch.inference_mode(), autocast(

            device_type="cuda",

            dtype=torch.bfloat16
        ):

            generated = summ_model.generate(

                **inputs,

                max_length=128,

                min_length=20,

                num_beams=6,

                num_beam_groups=3,

                diversity_penalty=0.5,

                num_return_sequences=4,

                repetition_penalty=1.2,

                no_repeat_ngram_size=3,

                length_penalty=1.0,

                early_stopping=True
            )

        decoded_candidates = summ_tokenizer.batch_decode(

            generated,

            skip_special_tokens=True
        )

        # =================================================
        # PROCESS EACH SAMPLE
        # =================================================
        for j in range(len(texts)):

            doc_text = texts[j]

            reference = refs[j]

            candidates = decoded_candidates[
                j * 4:(j + 1) * 4
            ]

            best_candidate = candidates[0]

            best_score = -999

            sample_candidate_logs = []

            # =============================================
            # REFERENCE EMBEDDING
            # =============================================
            ref_embedding = sim_model.encode(

                reference,

                convert_to_tensor=True
            )

            for cand in candidates:

                try:

                    # =====================================
                    # NLI SCORE
                    # =====================================
                    nli_inputs = nli_tokenizer(

                        doc_text,

                        cand,

                        return_tensors="pt",

                        truncation=True,

                        max_length=512

                    ).to(device)

                    with torch.inference_mode(), autocast(

                        device_type="cuda",

                        dtype=torch.bfloat16
                    ):

                        logits = nli_model(
                            **nli_inputs
                        ).logits

                        probs = torch.softmax(
                            logits,
                            dim=-1
                        )

                        entailment_score = probs[
                            0,
                            ENTAILMENT_IDX
                        ].item()

                    # =====================================
                    # SEMANTIC SCORE
                    # =====================================
                    cand_embedding = sim_model.encode(

                        cand,

                        convert_to_tensor=True
                    )

                    semantic_score = util.cos_sim(

                        cand_embedding,

                        ref_embedding

                    ).item()

                    # =====================================
                    # COMET SCORE
                    # =====================================
                    comet_input = [{
                        "src": doc_text,
                        "mt": cand,
                        "ref": reference
                    }]

                    comet_output = comet_model.predict(
                        comet_input,
                        batch_size=1,
                        gpus=1 if torch.cuda.is_available() else 0
                    )

                    comet_score = float(
                        comet_output.scores[0]
                    )

                    # =====================================
                    # LENGTH SCORE
                    # =====================================
                    cand_len = len(
                        cand.split()
                    )

                    length_score = min(
                        cand_len / 60.0,
                        1.0
                    )

                    # =====================================
                    # FINAL MULTI-OBJECTIVE SCORE
                    # =====================================
                    final_score = (

                        0.55 * entailment_score +

                        0.25 * semantic_score +

                        0.15 * comet_score +

                        0.05 * length_score
                    )

                    all_entailment_scores.append(
                        entailment_score
                    )

                    all_semantic_scores.append(
                        semantic_score
                    )

                    all_comet_scores.append(
                        comet_score
                    )

                    sample_candidate_logs.append({

                        "candidate": cand,

                        "entailment_score":
                            entailment_score,

                        "semantic_score":
                            semantic_score,

                        "comet_score":
                            comet_score,

                        "length_score":
                            length_score,

                        "final_score":
                            final_score
                    })

                    if final_score > best_score:

                        best_score = final_score

                        best_candidate = cand

                except Exception as inner_e:

                    print(f"⚠️ Candidate Error: {inner_e}")

                    continue

            best_summaries.append(
                best_candidate
            )

            candidate_logs.append({

                "source":
                    doc_text[:300],

                "reference":
                    reference,

                "best_summary":
                    best_candidate,

                "candidate_scores":
                    sample_candidate_logs
            })

        # =================================================
        # GPU LOGGING
        # =================================================
        gpu_mem = round(
            torch.cuda.memory_allocated() / 1e9,
            4
        )

        gpu_logs.append({

            "step": i,

            "gpu_memory_gb": gpu_mem,

            "time": time.time()
        })

        progress_bar.set_postfix({

            "gpu_gb": gpu_mem,

            "processed": i
        })

        # =================================================
        # PERIODIC SAVING
        # =================================================
        if i % 50 == 0:

            partial_df = pd.DataFrame({

                "source":
                    source_texts,

                "reference":
                    ground_truths,

                "prediction":
                    best_summaries
            })

            partial_df.to_csv(

                ALL_PREDICTIONS_CSV,

                index=False
            )

            gpu_df = pd.DataFrame(gpu_logs)

            gpu_df.to_csv(

                GPU_LOG_CSV,

                index=False
            )

            torch.cuda.empty_cache()

            gc.collect()

    except RuntimeError:

        print(f"\n⚠️ Runtime Error:")

        print(traceback.format_exc())

        torch.cuda.empty_cache()

        gc.collect()

        continue

# =====================================================
# 11. FINAL METRICS
# =====================================================
print("\n📊 Computing metrics...")

rouge_results = rouge.compute(

    predictions=best_summaries,

    references=ground_truths,

    use_stemmer=True
)

bertscore_results = bertscore.compute(

    predictions=best_summaries,

    references=ground_truths,

    lang="en"
)

avg_bertscore_f1 = np.mean(

    bertscore_results["f1"]
)

avg_entailment = np.mean(
    all_entailment_scores
)

avg_semantic = np.mean(
    all_semantic_scores
)

avg_comet = np.mean(
    all_comet_scores
)

# =====================================================
# 12. LANGUAGE PAIRWISE METRICS
# =====================================================
pairwise_rows.append({

    "source_language": "english",

    "target_languages":
        "spanish/french/turkish/swahili/portuguese",

    "rouge1":
        round(rouge_results["rouge1"] * 100, 4),

    "rouge2":
        round(rouge_results["rouge2"] * 100, 4),

    "rougeL":
        round(rouge_results["rougeL"] * 100, 4),

    "rougeLsum":
        round(rouge_results["rougeLsum"] * 100, 4),

    "bertscore_f1":
        round(avg_bertscore_f1 * 100, 4),

    "avg_entailment":
        round(avg_entailment * 100, 4),

    "avg_semantic_similarity":
        round(avg_semantic * 100, 4),

    "avg_comet":
        round(avg_comet * 100, 4)
})

pairwise_df = pd.DataFrame(pairwise_rows)

pairwise_df.to_csv(

    PAIRWISE_METRICS_CSV,

    index=False
)

# =====================================================
# 13. FINAL RESULTS
# =====================================================
print("\n" + "=" * 90)
print("🏆 FINAL RESULTS")
print("=" * 90)

final_metrics = {}

for key, value in rouge_results.items():

    score = round(value * 100, 4)

    final_metrics[key] = score

    print(f"{key}: {score}")

final_metrics["bertscore_f1"] = round(

    avg_bertscore_f1 * 100,

    4
)

final_metrics["avg_entailment"] = round(
    avg_entailment * 100,
    4
)

final_metrics["avg_semantic_similarity"] = round(
    avg_semantic * 100,
    4
)

final_metrics["avg_comet"] = round(
    avg_comet * 100,
    4
)

print(
    f"BERTScore-F1: {final_metrics['bertscore_f1']}"
)

print(
    f"Avg Entailment: {final_metrics['avg_entailment']}"
)

print(
    f"Avg Semantic Similarity: {final_metrics['avg_semantic_similarity']}"
)

print(
    f"Avg COMET: {final_metrics['avg_comet']}"
)

# =====================================================
# 14. SAVE METRICS
# =====================================================
with open(
    FINAL_METRICS_JSON,
    "w"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )

# =====================================================
# 15. SAVE QUALITATIVE EXAMPLES
# =====================================================
qualitative_df = pd.DataFrame({

    "Source":
        source_texts[:50],

    "Reference":
        ground_truths[:50],

    "Prediction":
        best_summaries[:50]
})

qualitative_df.to_csv(

    QUALITATIVE_CSV,

    index=False
)

# =====================================================
# 16. SAVE ALL PREDICTIONS
# =====================================================
all_predictions_df = pd.DataFrame({

    "source":
        source_texts,

    "reference":
        ground_truths,

    "prediction":
        best_summaries
})

all_predictions_df.to_csv(

    ALL_PREDICTIONS_CSV,

    index=False
)

# =====================================================
# 17. SAVE CANDIDATE SCORES
# =====================================================
candidate_rows = []

for sample in candidate_logs:

    for cand in sample["candidate_scores"]:

        row = {

            "source":
                sample["source"],

            "reference":
                sample["reference"],

            "best_summary":
                sample["best_summary"],

            "candidate":
                cand["candidate"],

            "entailment_score":
                cand["entailment_score"],

            "semantic_score":
                cand["semantic_score"],

            "comet_score":
                cand["comet_score"],

            "length_score":
                cand["length_score"],

            "final_score":
                cand["final_score"]
        }

        candidate_rows.append(row)

candidate_df = pd.DataFrame(candidate_rows)

candidate_df.to_csv(

    CANDIDATE_CSV,

    index=False
)

# =====================================================
# 18. SAVE GPU LOGS
# =====================================================
gpu_df = pd.DataFrame(gpu_logs)

gpu_df.to_csv(

    GPU_LOG_CSV,

    index=False
)

# =====================================================
# 19. SAVE JSON LOGS
# =====================================================
with open(
    CANDIDATE_JSON,
    "w"
) as f:

    json.dump(

        candidate_logs,

        f,

        indent=2,

        ensure_ascii=False
    )

# =====================================================
# 20. SAVE REPORT
# =====================================================
total_time_hours = round(

    (time.time() - start_time) / 3600,

    4
)

report = {

    "model_path":
        SUMM_MODEL_PATH,

    "test_samples":
        len(test_dataset),

    "runtime_hours":
        total_time_hours,

    "rouge1":
        final_metrics["rouge1"],

    "rouge2":
        final_metrics["rouge2"],

    "rougeL":
        final_metrics["rougeL"],

    "rougeLsum":
        final_metrics["rougeLsum"],

    "bertscore_f1":
        final_metrics["bertscore_f1"],

    "avg_entailment":
        final_metrics["avg_entailment"],

    "avg_semantic_similarity":
        final_metrics["avg_semantic_similarity"],

    "avg_comet":
        final_metrics["avg_comet"]
}

with open(
    FINAL_REPORT,
    "w"
) as f:

    for k, v in report.items():

        f.write(f"{k}: {v}\n")

# =====================================================
# 21. FINAL SUMMARY
# =====================================================
print("\n" + "=" * 90)
print("🎉 EMNLP PIPELINE COMPLETE")
print("=" * 90)

print(f"\n✅ Final Metrics:")
print(FINAL_METRICS_JSON)

print(f"\n✅ Pairwise Metrics:")
print(PAIRWISE_METRICS_CSV)

print(f"\n✅ Qualitative Examples:")
print(QUALITATIVE_CSV)

print(f"\n✅ All Predictions:")
print(ALL_PREDICTIONS_CSV)

print(f"\n✅ Candidate Scores:")
print(CANDIDATE_CSV)

print(f"\n✅ GPU Logs:")
print(GPU_LOG_CSV)

print(f"\n✅ Candidate JSON:")
print(CANDIDATE_JSON)

print(f"\n✅ Evaluation Report:")
print(FINAL_REPORT)

print("\n🚀 READY FOR EMNLP PAPER TABLES + GRAPHS")