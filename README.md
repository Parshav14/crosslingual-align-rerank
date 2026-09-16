# Cross-Lingual Semantic Alignment and Multi-Objective Reranking for Multilingual Abstractive Summarization

Code, results and figures for our paper in *Knowledge-Based Systems* (2026).

**Paper:** https://doi.org/10.1016/j.knosys.2026.116785

Parshav Singla, Jasmeet Singh, Gurpal Singh Chhabra
Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology

## The problem

Give a summarization model a news article in English and it writes a good summary. Give it the same article in Swahili and it writes a good summary of what appears to be a different article.

Both summaries are fluent. Both are confident. They cannot both be right.

Multilingual models are very good at sounding correct in many languages. This project is about making them agree with each other, and with the source, while they do it.

## What we did

The pipeline has three stages.

**1. Fine-tune.** mT5-Large is fine-tuned on XL-Sum in English, French, Spanish, Turkish and Swahili, 131,881 article and summary pairs in total. English is capped at 50,000 samples so it does not take over the conversation, which English tends to do.

**2. Align.** Articles are translated with a frozen M2M100 model. We then train the encoder so that an article and its translation end up in the same place in representation space, using a cosine alignment loss. Only the encoder is updated. The decoder already knows how to write, and in our ablation, retraining it bought almost no extra consistency and cost 2.57 ROUGE-L points. So we left it alone.

**3. Rerank.** For each article the model samples candidate summaries. Each one is scored on three things:

| Score | Model | Weight |
|---|---|---|
| Is it supported by the source? | mDeBERTa-v3 NLI | 0.55 |
| Is it about the source? | Multilingual MPNet | 0.30 |
| Does it agree with the other candidates? | Multilingual MPNet | 0.15 |

The highest scoring candidate wins. The reranker needs no training. It is a panel of judges, not another student.

## Results

Averages across all five languages (paper Table 7):

| Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | Consistency |
|---|---|---|---|---|
| Fine-tuning | 36.39 | 16.79 | 29.12 | 0.4468 |
| + Semantic alignment | 35.50 | 15.64 | 27.99 | 0.5263 |
| + Reranking (full framework) | 33.28 | 13.05 | 24.93 | 0.6710 |

Pairwise consistency before and after alignment (paper Table 3):

| Pair | Before | After | Gain |
|---|---|---|---|
| English to French | 0.4979 | 0.5416 | +0.0437 |
| English to Spanish | 0.5186 | 0.5767 | +0.0581 |
| English to Turkish | 0.4601 | 0.5128 | +0.0527 |
| English to Swahili | 0.3106 | 0.4740 | +0.1634 |

Yes, ROUGE goes down while consistency goes up. That is the point. ROUGE counts shared words with the reference. It does not check whether those words are true. Across 2,500 generated summaries, ROUGE-L and NLI entailment correlate at r = 0.069, which in statistical terms means they have met once and did not exchange numbers.

## Repository layout

```
crosslingual-align-rerank/
  README.md
  requirements.txt
  code/
    pipeline_multilingual_summarization.ipynb
  outputs/        CSV results behind the paper tables
  figures/        Figures from the paper
  archive/
    emnlp-prototype/
      NOTES.md    Read this before opening anything else in this folder
```

## The notebook

`code/pipeline_multilingual_summarization.ipynb` is the full working notebook, cell by cell, in the order it was run.

| Notebook section | What it does | Paper |
|---|---|---|
| Step 1 | Load XL-Sum, cap English at 50,000, fine-tune mT5-Large (lr 1.5e-5, 3 epochs, effective batch 16, label smoothing 0.1), evaluate on the test sets | Sections 3.1, 4.2, Table 2 |
| Steps 2 to 4 | M2M100 translation, MPNet consistency scoring, consistency before alignment | Section 3.2, Table 3 |
| Step 5 | Translate training articles, then encoder-only alignment (alpha 0.15, lr 5e-6, batch 2, 1 epoch, decoder frozen) | Sections 3.2, 4.2 |
| Step 6 | ROUGE and consistency after alignment | Tables 3 and 7 |
| Step 7 | Candidate generation and reranking | Sections 3.3, 3.4 |

Things you should know before running it:

- **You need a big GPU.** It was run on an NVIDIA H100. A laptop will politely decline.
- **Paths are hardcoded** to our training machine (`/workspace/jasmeet/...`). Change them to yours.
- **The Step 7 cell is an earlier configuration of the reranker.** It samples 8 candidates and scores them with NLI and centrality only (0.55 and 0.45). The configuration reported in the paper uses 3 candidates, nucleus sampling with p = 0.88, and the three weights in the table above. Use the paper's settings if you are reproducing Table 5.
- **Trained checkpoints are not included.** They are several gigabytes each, and they did not survive the end of our GPU allocation. We have made our peace with this.

## Data

We use [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) (Hasan et al., 2021), released under CC BY-NC-SA 4.0. The dataset is not included here. It is large, it is already hosted, and it belongs to its authors.

## Citation

```bibtex
@article{singla2026crosslingual,
  title   = {Cross-lingual semantic alignment and multi-objective reranking for multilingual abstractive summarization},
  author  = {Singla, Parshav and Singh, Jasmeet and Chhabra, Gurpal Singh},
  journal = {Knowledge-Based Systems},
  volume  = {351},
  pages   = {116785},
  year    = {2026},
  doi     = {10.1016/j.knosys.2026.116785}
}
```

## Contact

Dr. Jasmeet Singh, jasmeet.singh@thapar.edu
