```markdown
# Cross-Lingual Semantic Alignment and Multi-Objective Reranking for Multilingual Abstractive Summarization

Code, results and figures for our paper in *Knowledge-Based Systems* (2026).

**Paper:** https://doi.org/10.1016/j.knosys.2026.116785

Parshav Singla, Jasmeet Singh, Gurpal Singh Chhabra
Thapar Institute of Engineering and Technology

## The Problem

Ask a summarization model to summarize a news article in English, and you get a good summary. Give it the same article in Swahili, and you may get a summary that says something different.

Both summaries read well. Only one can be right.

This project makes the model give the same correct summary, no matter which language the article is in.

## How It Works

**Input:** a news article in English, French, Spanish, Turkish or Swahili.

**Output:** a short summary in the same language that sticks to the facts in the article.

The system works in three steps.

**1. Learn to summarize**
We train mT5-Large, a large multilingual model, on thousands of BBC news articles and their summaries in five languages.

**2. Learn that translations mean the same thing**
We translate articles into other languages and teach the model that an article and its translation carry the same meaning. Only the part of the model that reads the article is updated. The part that writes the summary is left untouched, so it keeps writing well.

**3. Pick the best summary**
The model writes a few different summaries for each article. Each one is checked for three things:

| Check | Weight |
|---|---|
| Does the article actually support what the summary says? | 0.55 |
| Is the summary about the article? | 0.30 |
| Does it agree with the other summaries? | 0.15 |

The summary with the highest score is chosen.

## Results

Averaged across all five languages:

| Stage | ROUGE-1 | ROUGE-2 | ROUGE-L | Consistency |
|---|---|---|---|---|
| Summarize | 36.39 | 16.79 | 29.12 | 0.4468 |
| + Translation training | 35.50 | 15.64 | 27.99 | 0.5263 |
| + Best summary selection | 33.28 | 13.05 | 24.93 | 0.6710 |

How much more consistent summaries became across languages:

| Languages | Before | After | Improvement |
|---|---|---|---|
| English and French | 0.4979 | 0.5416 | +0.0437 |
| English and Spanish | 0.5186 | 0.5767 | +0.0581 |
| English and Turkish | 0.4601 | 0.5128 | +0.0527 |
| English and Swahili | 0.3106 | 0.4740 | +0.1634 |

**Why ROUGE goes down:** ROUGE only counts how many words a summary shares with a reference summary. It cannot tell whether those words are true. Our system prefers a summary that is correct over one that simply copies the right words. So ROUGE drops a little, while consistency and factual accuracy go up.

Swahili gained the most. It has the least training data, so it had the most room to improve.

## Repository

```
crosslingual-align-rerank/
  code/       Full pipeline notebook
  outputs/    Result files behind the tables above
  figures/    Figures from the paper
```

## Data

We use [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum), a public dataset of BBC news articles and summaries (CC BY-NC-SA 4.0). It is not included here and can be downloaded from Hugging Face.

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
```
