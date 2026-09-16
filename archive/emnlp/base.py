# =====================================================
# EMNLP BASELINE EVALUATION PIPELINE
# Multilingual Baseline Comparison
# =====================================================

import os
import gc
import json
import torch
import pandas as pd
import numpy as np
import evaluate

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from sentence_transformers import (
    SentenceTransformer,
    util
)

# =====================================================
# GPU + MEMORY SETTINGS
# =====================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gc.collect()
torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🚀 DEVICE: {DEVICE}")

# =====================================================
# INPUT FILE
# SAME 1000-SAMPLE EVALUATION SET
# =====================================================

INPUT_FILE = "/workspace/jasmeet/entailment_contrastive/final_results/csv_logs/all_predictions_with_languages.csv"

df = pd.read_csv(INPUT_FILE)

print(f"\n✅ Loaded {len(df)} evaluation samples.")

# =====================================================
# OUTPUT DIRECTORY
# =====================================================

OUTPUT_DIR = "/workspace/jasmeet/entailment_contrastive/baseline_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_DIR = os.path.join(OUTPUT_DIR, "csv_logs")
JSON_DIR = os.path.join(OUTPUT_DIR, "json_logs")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# =====================================================
# BASELINE MODELS
# =====================================================

MODELS = {

    "mT5-Large":
    "google/mt5-large",

    "mBART-50":
    "facebook/mbart-large-50-many-to-many-mmt",

    "FLAN-T5-Large":
    "google/flan-t5-large",

    "PEGASUS-XSum":
    "google/pegasus-xsum"
}

# =====================================================
# METRICS
# =====================================================

print("\n📥 Loading Metrics...")

rouge = evaluate.load("rouge")

bertscore = evaluate.load("bertscore")

semantic_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# =====================================================
# LANGUAGE LIST
# =====================================================

languages = sorted(
    df["language"].dropna().unique()
)

# =====================================================
# FINAL RESULTS
# =====================================================

all_results = []

# =====================================================
# LOOP OVER MODELS
# =====================================================

for model_name, model_path in MODELS.items():

    print("\n" + "="*90)
    print(f"🚀 RUNNING BASELINE: {model_name}")
    print("="*90)

    gc.collect()
    torch.cuda.empty_cache()

    # =================================================
    # LOAD MODEL
    # =================================================

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    ).to(DEVICE)

    model.eval()

    # =================================================
    # STORE ALL PREDICTIONS
    # =================================================

    model_predictions = []

    # =================================================
    # LANGUAGE LOOP
    # =================================================

    for lang in languages:

        print("\n" + "-"*70)
        print(f"🌍 LANGUAGE: {lang}")
        print("-"*70)

        subdf = df[
            df["language"] == lang
        ].copy()

        sources = subdf["source"].astype(str).tolist()

        references = subdf["reference"].astype(str).tolist()

        predictions = []

        batch_size = 4

        # =============================================
        # GENERATION LOOP
        # =============================================

        for i in tqdm(

            range(0, len(sources), batch_size),

            desc=f"{model_name} | {lang}"
        ):

            batch_texts = sources[
                i:i+batch_size
            ]

            inputs = tokenizer(

                batch_texts,

                padding=True,
                truncation=True,

                max_length=512,

                return_tensors="pt"

            ).to(DEVICE)

            with torch.no_grad():

                outputs = model.generate(

                    **inputs,

                    max_length=128,

                    min_length=15,

                    num_beams=4,

                    length_penalty=1.0,

                    no_repeat_ngram_size=3,

                    early_stopping=True
                )

            decoded = tokenizer.batch_decode(

                outputs,

                skip_special_tokens=True
            )

            predictions.extend(decoded)

            gc.collect()
            torch.cuda.empty_cache()

        # =============================================
        # CLEAN
        # =============================================

        predictions = [

            p.replace("<extra_id_0>", "").strip()

            for p in predictions
        ]

        # =============================================
        # SAVE RAW PREDICTIONS
        # =============================================

        lang_pred_df = pd.DataFrame({

            "language": lang,

            "source": sources,

            "reference": references,

            "prediction": predictions
        })

        lang_save_path = os.path.join(

            CSV_DIR,

            f"{model_name}_{lang}_predictions.csv"
        )

        lang_pred_df.to_csv(

            lang_save_path,

            index=False
        )

        print(f"\n💾 Saved Predictions: {lang_save_path}")

        # =============================================
        # ROUGE
        # =============================================

        rouge_scores = rouge.compute(

            predictions=predictions,

            references=references
        )

        # =============================================
        # BERTSCORE
        # =============================================

        bert = bertscore.compute(

            predictions=predictions,

            references=references,

            lang="multi"
        )

        bert_f1 = np.mean(bert["f1"])

        # =============================================
        # SEMANTIC SIMILARITY
        # =============================================

        sims = []

        for pred, ref in tqdm(

            zip(predictions, references),

            total=len(predictions),

            desc=f"Semantic {lang}"
        ):

            emb1 = semantic_model.encode(

                pred,

                convert_to_tensor=True
            )

            emb2 = semantic_model.encode(

                ref,

                convert_to_tensor=True
            )

            sim = util.cos_sim(

                emb1,

                emb2

            ).item()

            sims.append(sim)

        semantic_score = np.mean(sims)

        # =============================================
        # STORE FINAL RESULTS
        # =============================================

        all_results.append({

            "Model": model_name,

            "Language": lang,

            "Samples": len(subdf),

            "ROUGE-1": round(
                rouge_scores["rouge1"] * 100,
                2
            ),

            "ROUGE-2": round(
                rouge_scores["rouge2"] * 100,
                2
            ),

            "ROUGE-L": round(
                rouge_scores["rougeL"] * 100,
                2
            ),

            "ROUGE-Lsum": round(
                rouge_scores["rougeLsum"] * 100,
                2
            ),

            "BERTScore-F1": round(
                bert_f1 * 100,
                2
            ),

            "SemanticSimilarity": round(
                semantic_score * 100,
                2
            )
        })

        print("\n✅ METRICS")

        print(f"ROUGE-1: {rouge_scores['rouge1']*100:.2f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']*100:.2f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']*100:.2f}")

        print(f"BERTScore: {bert_f1*100:.2f}")
        print(f"Semantic Similarity: {semantic_score*100:.2f}")

    # =================================================
    # FREE MEMORY
    # =================================================

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

# =====================================================
# FINAL SAVE
# =====================================================

results_df = pd.DataFrame(all_results)

final_csv = os.path.join(

    CSV_DIR,

    "all_baseline_results.csv"
)

results_df.to_csv(

    final_csv,

    index=False
)

# =====================================================
# JSON SAVE
# =====================================================

final_json = os.path.join(

    JSON_DIR,

    "all_baseline_results.json"
)

with open(final_json, "w") as f:

    json.dump(

        all_results,

        f,

        indent=4
    )

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n" + "="*90)
print("🏆 ALL BASELINE EVALUATION COMPLETE")
print("="*90)

print("\n📄 FINAL CSV:")
print(final_csv)

print("\n📄 FINAL JSON:")
print(final_json)

print("\n✅ READY FOR EMNLP TABLES + FIGURES")