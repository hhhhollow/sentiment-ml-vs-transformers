# Data Card: UCI Sentiment Labelled Sentences

## Provenance and license

- Canonical source: https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip
- UCI DOI: https://doi.org/10.24432/C57604
- License: CC BY 4.0; attribution to Dimitrios Kotzias and the UCI Machine Learning
  Repository is required.
- Pinned archive SHA-256: `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`

The corpus contains English sentences sampled from product, movie, and restaurant reviews. UCI
reports 500 positive and 500 negative sentences per source and says neutral sentences were excluded.

## Processing and split

The parser reads physical lines, strips surrounding whitespace, and separates the label at the final
tab. It verifies the archive
and all three member hashes, validates binary labels and non-empty text, then removes duplicate
normalised-text keys globally before any split. This produces 2,979 rows from
3,000 source rows. The retained original sentence is used for modelling; the normalised
key exists only for duplicate/leakage control.

Stable SHA-256-derived sentence IDs replace row positions. A source×label-stratified split assigns
1,787 train, 596 validation, and 596 test
rows. The mapping is persisted, and ID/text overlap checks fail closed.

## Known limitations

Labels encode deliberately clear binary sentiment rather than the full ambiguity of natural
language. The dataset is small, English-only, and sourced from three older review domains. It
contains no demographic labels, timestamp, annotator agreement, or reliable author identity. It is
appropriate for an educational model-comparison benchmark, not for claims about production drift,
multilinguality, individual people, or demographic fairness.
