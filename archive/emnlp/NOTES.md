# Earlier prototype

This folder holds an earlier version of the project, built before the final pipeline in `code/`. It is kept for history. It does not reproduce the numbers in the paper.

## How it differs from the paper

| | This prototype | Published paper |
|---|---|---|
| Languages | English, French, Spanish, Portuguese, Turkish, Swahili | English, French, Spanish, Turkish, Swahili |
| Training data | Filtered by NLI entailment (threshold 0.5), about 6% kept | Full training splits, English capped at 50,000 |
| Extra loss | Contrastive and consistency terms on decoder states | Cosine alignment on mean-pooled encoder states of translated pairs |
| Translation model | Not used | M2M100 |
| Candidates | Diverse beam search, 4 candidates | Nucleus sampling, 3 candidates |
| Reranking score | 0.55 NLI, 0.25 similarity, 0.15 COMET, 0.05 length | 0.55 NLI, 0.30 source relevance, 0.15 centrality |
| Baselines (`base.py`) | Pretrained models, zero-shot | Fine-tuned mBART-50 and published baselines |

## Known issue

In `02_train.py`, `pos_sim` compares `pooled` with `pooled.detach()`. That is a vector compared with itself, so the similarity is always 1. The consistency loss is therefore always about zero. We found this later and left the file unchanged so it matches what was actually run.
