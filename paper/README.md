# Manuscript source

`arxiv_v1.tex` and `arxiv_v1.bbl` are the source files of
[arXiv:2609.29387](https://arxiv.org/abs/2609.29387);
`scripts/tables/published_tables.py` writes the 19 tables from them.

`arxiv_source_manifest.json` is arXiv's source-processing metadata.
`SHA256SUMS` records hashes for the manuscript and the four figures. Run it from the repository root with:

```bash
sha256sum --check paper/SHA256SUMS
```

The manuscript and its figures are distributed under the arXiv terms; the
Apache license of the code does not apply to them.
