# ProteoMeter PTM reference

This reference supplements `SKILL.md` with repository-specific details. Prefer the local source files over copied snippets if the package changes.

This reference covers the typical ProteoMeter PTM workflow, not a general proteomics schema. Do not reuse its PTM input assumptions for LiP or other analysis modes without reviewing their dedicated configs and APIs.

For the full repository documentation entry point, see [docs/index.rst](../../../docs/index.rst). Use the source modules under `../../../src/proteometer/` when the focused PTM reference does not cover an API.

## Input tables

The demo uses tab-separated files under `demo_data/PTM/`:

- `global_prot.tsv`: global protein quantification.
- `global_pept.tsv`: global peptide quantification.
- `acetyl_pept.tsv`, `phospho_pept.tsv`, and `redox_pept.tsv`: one modified-peptide table per PTM type.
- `metadata.tsv`: sample design and batch mapping.
- `reference_proteome.fasta`: FASTA used for matching and optional iBAQ.

For PTM peptide tables, the default config expects `UniProt`, `Protein`, `Peptide`, and `Residue`, plus numeric columns named exactly like metadata samples. The modified peptide sequence must use the configured PTM symbol convention. The package creates peptide/site identifiers from UniProt and peptide/residue information, so identifiers should be stable and string-like.

Global and PTM tables must share the relevant sample columns. Global tables are not optional when abundance correction is enabled. The FASTA is also not optional when `ibaq = true`.

These are post-search, standardized TSV inputs. The PTM pipeline does not automatically normalize arbitrary raw MaxQuant, FragPipe, or MSFragger exports into this schema; perform and document that mapping separately.

## Metadata contract

The demo config maps these names:

```text
metadata_batch_col       = Batch
metadata_sample_col      = Sample
metadata_group_col       = Group
metadata_condition_col   = Condition
metadata_control_condition   = Control
metadata_treatment_condition = Treatment
```

The metadata file should contain one row per sample. Typical design columns are:

```text
Sample  Group  Condition  Replicate  Batch
```

Additional columns can encode treatment, time, dose, cell line, or other factors. If a factor is placed in `statistics.anova_factors`, its values must be present for every analyzed sample.

The `Sample` values in metadata are the join contract: they must match intensity column headers exactly, including case and punctuation. The demo's group/condition distinction is intentional:

- `Group`: comparison units such as `Mock_8h` and `Infected_8h`.
- `Condition`: role labels such as `Control`, `Treatment`, and pooled `Total`.

Pooled channels should be handled consistently with `pooled_chanel_condition` and excluded from unsuitable ANOVA comparisons by the package metadata parser.

## Configuration invariants

`Params` validates these values while loading TOML:

- `experiment.experiment_type` is exactly `TMT` or `Label-free`.
- `experiment.lip.search_tool` is `maxquant`, `msfragger`, `fragpipe`, or `""`; PTM-only configs normally use `""`.
- `corrections.sig_type` is `pval` or `adj-p`.
- The lengths of `ptm_names`, `ptm_pept_files`, `ptm_symbols`, and `ptm_abbreviations` are equal.

Paths are resolved relative to the current working directory by `Params`, then stored as absolute paths. Prefer running from the repository root or use paths whose base is unambiguous. `results_dir` is created by `Params`.

The demo's key PTM configuration looks like this conceptually:

```toml
[paths]
data_dir = "./demo_data/PTM"
results_dir = "./demo_data/"
fasta_file = "reference_proteome.fasta"
metadata_file = "metadata.tsv"
global_prot_file = "global_prot.tsv"
global_pept_file = "global_pept.tsv"

[paths.ptm]
ptm_pept_files = ["acetyl_pept.tsv", "phospho_pept.tsv", "redox_pept.tsv"]

[experiment]
experiment_type = "TMT"

[statistics]
ttest_pairs = [["Mock_8h", "Infected_8h"]]

[symbols.ptm]
ptm_names = ["acetyl", "phospho", "redox"]
ptm_symbols = ["@", "#", "@"]
ptm_abbreviations = ["Ac", "Ph", "Ox"]
```

Use the complete `demo_data/ptm.toml` as the template because correction, metadata, and output-column settings are also required by the implementation.

For downstream code, prefer the package's metadata helpers instead of inferring sample columns manually:

```python
from proteometer import parse_metadata

metadata = pd.read_csv(params.metadata_file, sep="\t")
intensity_columns = parse_metadata.int_columns(metadata, params)
group_columns, group_names = parse_metadata.group_columns(metadata, params)
```

## Pipeline and return values

`ptm_analysis(params, drop_samples=None)` performs, in order:

1. Read metadata and all quantification tables.
2. Cast protein and UniProt identifiers to strings.
3. Parse intensity columns, groups, ANOVA columns, and t-test groups from metadata.
4. Generate identifiers and optionally log2-transform intensity values.
5. Filter missingness by group.
6. Normalize global protein data and calculate its statistics first.
7. Normalize modified peptides against global peptides.
8. Roll peptides up to residue/site-level data.
9. Optionally batch-correct and abundance-correct site values.
10. Calculate site-level ANOVA and pairwise t-test results.
11. Combine all PTM types and global protein data.
12. Apply final missingness filtering and optional FASTA-based iBAQ.

It returns `(all_ptms, global_prot)`. `all_ptms` contains combined site-level PTM rows and `global_prot` contains processed global protein rows. `ptm_analysis_return_all(params)` additionally returns uncorrected PTM site results as the third value.

The output includes configured identifier columns and generated fields such as `type`, `experiment`, `site_number`, comparison effect columns, and p-value/adjusted-p-value columns. Exact comparison prefixes come from `TTestGroup.label()`, so inspect `params.ttest_pairs` and `df.columns` rather than hard-coding names.

## Post-analysis recipes

Use the patterns in `demonstration.ipynb` as the working example:

```python
import matplotlib.pyplot as plt
import proteometer.quality_control_plots as qcp

fig, ax = plt.subplots()
qcp.correlation_plot(ptm_site, intensity_columns, ax=ax)

fig, ax = plt.subplots()
qcp.biplot(ptm_site.dropna(), intensity_columns, grouped_sample_columns, ax=ax)

fig, ax = plt.subplots()
qcp.volcano_plot(ptm_site, comparison, ax=ax, sig_type="adj-p", sig_thresh=0.05)
```

For multiple-testing review:

```python
from proteometer.stats import recalculate_adj_pval, recalculate_adj_pval_proteinwise

site_global_fdr = recalculate_adj_pval(ptm_site.copy(), comparisons)
site_proteinwise_fdr = recalculate_adj_pval_proteinwise(
    ptm_site.copy(), comparisons, protein_col=params.protein_col
)
```

Use the package's FASTA and barcode/alignment utilities when the question concerns sequence context, peptide coverage, or residue localization. For enrichment, the demo uses `gseapy` and an externally downloaded Hallmark GMT file; record the GMT source/version and do not imply enrichment is part of the core `ptm_analysis` call.

## Diagnostics checklist

- `FileNotFoundError`: print the absolute paths held by `Params`; check `data_dir` and the current working directory.
- PTM list `ValueError`: compare all four PTM lists element by element.
- `KeyError` for a sample: compare metadata sample values with every table's intensity headers.
- Empty or heavily filtered output: inspect missingness and whether `min_replicates_qc` is too strict for the design.
- Missing statistics: verify `ttest_pairs`, exact group names, and replicate counts after exclusions.
- Batch correction failure: verify every configured `batch_correct_samples` value exists in metadata and that the batch column is populated.
- Abundance correction failure or implausible values: confirm global tables are from paired samples when configured, and inspect global protein statistics before interpreting PTM changes.
- iBAQ/FASTA mismatch: inspect FASTA headers and choose the appropriate matching mode; report unmatched identifiers.
- Residue parsing failure: compare residue strings and modified peptide notation against the demo input files and `src/proteometer/residue.py`.

After every run, report row counts before/after filtering, analyzed samples, excluded samples, corrections enabled, significance settings, and the output files produced.

## Run summary artifact

Create the durable report from `../assets/ptm-analysis-summary.md` after the pipeline run, before optional post-analysis. Update that same Markdown file after approved QC and downstream analyses. The summary is complete only after it records the final outputs, design and configuration decisions, QC findings, post-analysis actions, caveats, and links to deeper documentation.
