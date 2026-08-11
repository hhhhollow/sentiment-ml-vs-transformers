# Model Cards

All three binary English-sentiment classifiers are educational artifacts. Their intended use is to
compare modelling approaches on the pinned UCI benchmark. They are not suitable for employment,
education, credit, health, moderation enforcement, or other consequential decisions about people.

| Model | Val Macro-F1 | Test Macro-F1 (95% CI) | Accuracy | ROC-AUC | Dev s | Infer ms/item | Params | Artifact MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DistilBERT | 0.9379 | 0.9228 [0.9025, 0.9428] | 0.9228 | 0.9711 | 200.15 | 8.526 | 66,955,010 | 256.33 |
| TF-IDF + Logistic Regression | 0.8456 | 0.8473 [0.8187, 0.8742] | 0.8473 | 0.9105 | 0.57 | 0.031 | 47,232 | 1.27 |
| PyTorch TextCNN | 0.7231 | 0.7164 [0.6810, 0.7514] | 0.7164 | 0.7989 | 6.61 | 0.160 | 296,961 | 1.21 |

## Selection and evaluation boundary

Every representation is fitted on train only. TF-IDF candidates and neural checkpoints are
selected by validation Macro-F1; the held-out test set never controls hyperparameters, epochs, or
the reported champion. Probabilities are converted with a fixed 0.50 threshold. Artifacts record
model-specific preprocessing so inference cannot silently use a different vocabulary or tokenizer.

## Artifact and inference contract

- Input: a non-blank English `text` string and an optional unique `sentence_id`.
- Output: positive-class probability, a label from a fixed 0.50 threshold, and the model name.
- Artifacts: `artifacts/models/tfidf_logistic.joblib`, `artifacts/models/textcnn/`, and
  `artifacts/models/distilbert/`. TextCNN and DistilBERT weights use safetensors; DistilBERT is
  derived from `distilbert/distilbert-base-uncased` at immutable revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- Probabilities are not post-hoc calibrated. Ten-bin ECE on this test split is diagnostic, not a
  guarantee for new domains:

- DistilBERT: test ECE = 0.0480
- TF-IDF + Logistic Regression: test ECE = 0.1639
- PyTorch TextCNN: test ECE = 0.0479

## Test Macro-F1 by source

| Model | Amazon | Imdb | Yelp |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8579 | 0.7949 | 0.8894 |
| PyTorch TextCNN | 0.7454 | 0.6750 | 0.7278 |
| DistilBERT | 0.9137 | 0.9200 | 0.9346 |

## Model-specific risks

- TF-IDF is sparse and efficient but weak at compositional meaning, negation over long spans, and
  unseen wording.
- TextCNN learns local n-gram-like patterns from only 1,787 training rows;
  variance and out-of-vocabulary behaviour are important.
- DistilBERT has much broader pretrained knowledge but inherits unknown pretraining biases and
  carries substantially more parameters. This benchmark excludes its upstream pretraining compute.

The bootstrap intervals quantify row-sampling uncertainty on one fixed test split. They do not
measure variance across alternative train/validation/test assignments or neural training seeds.
