# Transformer from Scratch: Finnish → English Machine Translation

Advanced NLP, Assignment 1 · Sanjana Sheela (2023102027)

## 1. Summary

I implemented an encoder–decoder Transformer in PyTorch with no `nn.Transformer*` modules and no pre-built attention, trained it on 100,000 Finnish–English sentence pairs from the EUbookshop corpus, and set it up to compare two positional-encoding schemes (rotary embeddings and a learned relative-position attention bias) under three decoding strategies (greedy, beam search, top-k sampling).

| Item | Status |
|---|---|
| Implementation | Complete: encoder, decoder, RoPE, relative bias, sinusoidal encoding, three decoders (§4–§7) |
| BLEU for each encoding × decoder | **To be filled in from this project's own runs** (§8.2); the notebook outputs in this folder were cleared |
| Convergence plot | To be generated from this project's own runs (§8.3) |
| Corpus quality | The data is noisy: about 15% of pairs have badly mismatched lengths and some are plainly misaligned (§3.4), which limits any model's BLEU |

BLEU in this project's evaluation code is computed over BPE tokens with NLTK, not over words with sacreBLEU, so scores are for comparing configurations here and should not be set against published numbers (§8.1).

## 2. Repository layout

| File | Role |
|---|---|
| [`encoder.py`](./encoder.py) | `EncoderBlock` and `Encoder` (self-attention + feed-forward, pre-norm residuals, final LayerNorm) |
| [`decoder.py`](./decoder.py) | `DecoderBlock` and `Decoder` (masked self-attention, cross-attention, feed-forward) |
| [`utils.py`](./utils.py) | Embeddings, sinusoidal / RoPE / relative-bias positional encodings, multi-head attention, LayerNorm, feed-forward, projection, dataset class, masks, tokenizer builder, greedy decoding |
| [`train.py`](./train.py) | Model assembly (`build_transformer`), data pipeline and splits, training loop, validation, metric plots, checkpointing |
| [`config.py`](./config.py) | Hyper-parameters, including the `positional_encoding` switch |
| [`runner.ipynb`](./runner.ipynb) | Kaggle launcher for training (see §10, it is partly out of date) |
| [`test_inference.ipynb`](./test_inference.ipynb) | Inference: beam search and top-k sampling, BLEU evaluation on the saved test split, custom sentences |
| `datasets/` | `EUbookshop.fi`, `EUbookshop.en`, a cached `custom_dataset.json`, and a 1,000-pair `small/` subset |
| `Assignment_1 (3).pdf` | The assignment brief |

Files created when training runs (not stored in this folder): `weights/tmodel_XX.pt`, `tokenizer_fi.json`, `tokenizer_en.json`, `dataset_splits/`, `plots/`, `runs/tmodel/` (TensorBoard).

The brief lists a `test.py` deliverable. There is none; evaluation lives in `test_inference.ipynb`.

## 3. Data

### 3.1 Corpus

EUbookshop Finnish–English parallel text (European Union publications), 100,000 aligned sentence pairs, read line by line from `datasets/EUbookshop.fi` and `.en` and cached as `custom_dataset.json`. Empty lines are dropped.

Measured on the files in `datasets/`:

| | Finnish | English |
|---|---|---|
| Mean words per sentence | 15.7 | 23.2 |
| Median / 90th / 99th percentile | 14 / 29 / 48 | 21 / 43 / 71 |
| Longest | 176 | 231 |

### 3.2 Splits

`random_split` into 80% train, 10% validation and 10% test (80,000 / 10,000 / 10,000 pairs before length filtering). Pairs whose tokenised source (+2 for `[SOS]`/`[EOS]`) or target (+1) exceeds `seq_len = 128` are removed from each split, and the retention rate is printed. The split indices, a metadata file, text samples and a pickle of the three partitions are written to `dataset_splits/` so the test notebook can reload the same test set.

### 3.3 Tokenisation

Separate BPE tokenisers for Finnish and English (Hugging Face `tokenizers`), whitespace pre-tokenisation, vocabulary size 3,000 each, `min_frequency=2`, and the special tokens `[UNK] [PAD] [SOS] [EOS]`. Tokenisers are trained once on the full corpus and then reloaded from `tokenizer_{lang}.json`.

### 3.4 Corpus quality

The data is noisier than the usual machine-translation benchmark. Checks on the 100,000 pairs:

| Check | Count |
|---|---|
| Pairs where the Finnish/English word-count ratio is above 2 or below 0.5 | 15,418 (15.4%) |
| Exact duplicate (Finnish, English) pairs | 4,477 (4.5%) |
| Finnish sentences of 2 words or fewer | 3,460 (3.5%) |
| Pairs with identical Finnish and English text | 0 |

Some pairs are not translations of each other. Two examples, copied from the data:

| Finnish | English |
|---|---|
| Puhemies. - Esityslistalla on seuraavana jatkokeskustelu neuvoston ja komission julkilausumista Albanian … | I would like to draw the presidency's attention to the fact that the massacre is particularly serious because it included women and children. |
| 20.06.1928 Euroopan parlamentin jäsen Ranska FN EPJ: 24.07.1984 NI | Member of the European Parliament France |

There are also hyphenation breaks and OCR slips ("sisäl-", "wUl", "deal ing"). Any model trained on this data has a ceiling on BLEU that comes from the references themselves.

## 4. Architecture

All blocks are written directly from `nn.Linear`, `nn.Embedding`, `nn.Dropout` and tensor operations.

| Setting | Value |
|---|---|
| Encoder / decoder layers | 2 / 2 |
| `d_model` | 512 |
| Attention heads | 2 (head size 256) |
| Feed-forward width | 1,024, ReLU |
| Dropout | 0.1 |
| Vocabulary | 3,000 source, 3,000 target (separate embeddings, no weight tying) |
| Sequence length | 128 |
| Initialisation | Xavier uniform for every weight matrix |
| Parameters | about 15.1 M (counted by hand from the settings above; the relative-bias tables add 260) |

The layer count, head count and feed-forward width are set inside `get_model()` in `train.py`, not in `config.py`.

Design details:

- **Pre-norm residual blocks:** `x + dropout(sublayer(LayerNorm(x)))`, with a final LayerNorm after each stack (`ResidualConnection`, `Encoder`, `Decoder`).
- **LayerNorm** is hand-written. Its learnable gain and bias are single scalars, not per-feature vectors as in the standard layer.
- **Embeddings** are scaled by √`d_model`.
- **Masks:** the source padding mask is used in encoder self-attention and decoder cross-attention. The decoder self-attention mask combines target padding with a causal (lower-triangular) mask.
- **Output layer:** linear projection followed by `log_softmax`.
- **Decoder attention:** self-attention applies the positional scheme; cross-attention applies none, since queries and keys come from different sequences.

## 5. Positional encodings

The scheme is chosen with `config['positional_encoding']`: `"rope"`, `"relative_bias"`, or `"vanilla"` (sinusoidal). Vanilla is implemented but was not part of the comparison.

### 5.1 Rotary position embeddings (RoPE)

`RotaryPositionalEmbedding` precomputes `cos` and `sin` tables for positions 0–127 with inverse frequencies `10000^(−2i/d_k)`, where `d_k` is the per-head dimension (256). Inside `MultiHeadAttention`, after the queries and keys are split into heads, each consecutive pair of dimensions is rotated by its position-dependent angle. Values are not rotated, and nothing is added to the embeddings. The attention score between positions *m* and *n* then depends on the relative offset *n − m*.

### 5.2 Relative position bias

`RelativePositionalBias` holds a learnable table of shape `[heads, 2·32 + 1]`, initialised from N(0, 0.02²). Relative offsets `j − i` are clamped to ±32 and used to look up one scalar per head, which is added to the attention scores before the mask and softmax. The same module instance is passed to every layer of a stack, so the encoder's layers share one bias table and the decoder's layers share another. Offsets beyond ±32 all share the same bias value.

## 6. Training

| Setting | Value |
|---|---|
| Objective | Teacher forcing, cross-entropy on shifted targets, label smoothing 0.1, padding ignored |
| Optimiser | Adam, learning rate 1e-3 (constant), eps 1e-9 |
| Batch size | 16 |
| Epochs | 15 |
| Warm-up, schedule, gradient clipping | None |
| Hardware | Kaggle GPUs (see the README) |

After every epoch the code runs greedy decoding on the first 100 validation sentences and logs NLTK BLEU, then saves `weights/tmodel_XX.pt` with model weights, optimiser state, epoch and global step (so `preload` can resume a run). Per-step loss goes to TensorBoard (`runs/tmodel`), and `MetricsTracker` writes loss and BLEU plots to `plots/`. The final model for evaluation is the last-epoch checkpoint.

## 7. Decoding strategies

| Strategy | Implementation | Settings in the evaluation helper |
|---|---|---|
| Greedy | `greedy_decode` in `utils.py`; takes the argmax token each step | none |
| Beam search | `beam_search_decode` in the notebook; keeps the top-B partial sequences by summed log-probability | beam size 2 |
| Top-k sampling | `top_k_sampling_decode` in the notebook; samples from the k most probable tokens after temperature scaling | k = 5, temperature 0.8 |

All three decode one sentence at a time and re-run the decoder over the whole prefix at every step (no key/value cache). Beam search has no length normalisation. The interactive helper `translate_text` uses different settings (beam size 5, k = 50), so the settings above are the ones in `evaluate_single_strategy_fast`, which is the function the BLEU evaluation calls.

## 8. Evaluation

### 8.1 Metric

Corpus BLEU from NLTK (`corpus_bleu`, default weights and no smoothing). NLTK returns values between 0 and 1; scores in this report are multiplied by 100. In the test notebook both prediction and reference are lists of **BPE tokens** (`id_to_token`), not whitespace words, and the in-training validation BLEU splits the decoded strings on spaces. BLEU over subword tokens and over words is not comparable, and neither equals sacreBLEU. The numbers below are useful for comparing the configurations in this project against each other.

### 8.2 BLEU

Fill this table from evaluation runs on one shared test split (BLEU × 100; the notebook writes `inference_results/bleu_scores_comparison.csv` when `run_all_strategies = True`). State the decoder settings used (§7) and the number of test sentences scored.

| Positional encoding | Greedy | Beam search | Top-k sampling |
|---|---|---|---|
| RoPE | | | |
| Relative position bias | | | |

The notebook outputs in this folder were cleared, so no scores are stored here.

### 8.3 Convergence

Plot training loss against epoch for both encodings on one graph and say which falls faster. The per-epoch loss comes from `MetricsTracker.plot_metrics()` (`plots/training_metrics_latest.png`) and from TensorBoard. A single figure with both encodings needs two runs that write to different `experiment_name` and `model_folder` values.

## 9. Analysis

This section covers what the code and the data support. Add the comparison of the two encodings and the three decoders once §8 is filled in.

**What differs between the two positional schemes.** RoPE rotates the queries and keys in every head (256 dimensions per head) before the dot product, so position enters each attention score directly and adds no parameters. The relative bias adds one learned scalar per head for each relative offset from −32 to +32, starts near zero, treats every offset beyond ±32 alike, and is shared by all layers of a stack. Any difference in BLEU or convergence between the two should be read against these design differences; nothing in the project ablates them.

**Decoding.** Beam search here has no length normalisation, and the evaluation helper uses a small beam (2), so it has little room to beat greedy decoding. Top-k sampling is stochastic, so its score can change between runs; report the seed or average several runs.

**Data.** The corpus limits the achievable score (§3.4): 15% of pairs have mismatched lengths, some are not translations of each other, and 4.5% are duplicates. The model learns from references that are sometimes wrong, and BLEU is then measured against references that are sometimes wrong too. The model is also small (2 layers, 2 heads, about 15 M parameters), and nothing in the project separates how much of a low score comes from the data and how much from the model.

## 10. Limitations and known issues

| Issue | Where | Effect |
|---|---|---|
| BLEU is computed over BPE tokens, unsmoothed | `test_inference.ipynb`, `train.py` | Scores are not comparable to word-level or sacreBLEU numbers |
| The test BLEU printed at the end of training uses only the first 100 test sentences | `run_validation(..., max_validation_samples=100)` in `train.py` | Noisy; use the notebook for the full test set |
| `random_split` has no seed, and the split is rebuilt on every call | `get_ds` in `train.py` | Two training runs get different splits unless the saved `dataset_splits/` is reused. If the RoPE and relative-bias models were evaluated on one test set but trained on different splits, one of them may have seen part of that test set |
| Exact duplicate pairs (4.5%) are not removed before splitting | `get_ds` | Some test sentences may also appear in the training data |
| Tokenisers are trained on the full corpus before splitting | `get_or_build_tokenizer` | Small vocabulary leakage from validation and test text |
| The `DataParallel` wrapper is bypassed, because the loop calls `model.module.encode/decode` directly | `train_model` | Training runs on one GPU even when several are visible; the "effective batch size" message is wrong |
| No learning-rate schedule, warm-up or gradient clipping | `train_model` | The constant 1e-3 rate is high for a Transformer and may limit convergence |
| Decoding is chosen by editing a notebook argument, not by command-line flag or config | `test_inference.ipynb` | The brief asks for switching through a CLI argument or a configuration file |
| Evaluation settings are inconsistent: `translate_text` uses beam 5 and k = 50, `evaluate_single_strategy_fast` uses beam 2 and k = 5 | `test_inference.ipynb` | Check which pair produced the reported scores |
| The BLEU-summary cell expects result keys `greedy / beam_search / top_k_sampling`, but the evaluation returns `top_k` or a single result | cell 20 of `test_inference.ipynb` | That cell raises a `KeyError` as written |
| `runner.ipynb` still imports `test_transformer`, refers to `test.py` and prints `use_rope` | `runner.ipynb` | Out of date; its saved output is an import error |
| The "best model" at the end of training is just the last epoch | `train_model` | No best-validation checkpoint is kept |
| Beam search and greedy decoding re-encode the full prefix at each step | decoders | Slow on the full test set |

## 11. Reproducing the results

Requirements: `torch`, `tokenizers`, `nltk`, `torchmetrics`, `tensorboard`, `matplotlib`, `tqdm`.

```bash
# from the project folder
python train.py
```

This builds the tokenisers, splits the data, trains for 15 epochs, runs a short validation after every epoch and writes `weights/`, `plots/`, `runs/` and `dataset_splits/`. For Kaggle, upload the `.py` files and `datasets/` and step through `runner.ipynb` (see the README).

To switch encodings, set `positional_encoding` in `config.py` to `"rope"` or `"relative_bias"`. A second run must also change `model_folder` (or `model_basename`) and `experiment_name`; otherwise it overwrites the first run's checkpoints and TensorBoard logs. To compare the two fairly, reuse the same tokenisers and the same `dataset_splits/` for both.

For inference, add the checkpoint (`tmodel_14.pt`), the saved splits and the tokenisers to Kaggle and run `test_inference.ipynb` cell by cell. Set `run_all_strategies = True` to evaluate all three decoders and save the comparison table.

