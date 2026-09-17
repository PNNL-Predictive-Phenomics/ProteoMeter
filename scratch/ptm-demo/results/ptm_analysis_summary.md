# ProteoMeter PTM Analysis Summary

## Run status

- Status: Final
- Date: 2026-09-16
- Analyst:
- Python interpreter: `/Users/geor228/Github/ProteoMeter/.venv/bin/python` (Python 3.12.9)
- ProteoMeter installation/import check: Passed; installed editable with uv
- Environment setup command: `uv pip install --python .venv/bin/python -e .`
- Summary path: `scratch/ptm-demo/results/ptm_analysis_summary.md`
- Configuration path: `scratch/ptm-demo/ptm.toml`
- Approval status: Approved

## Experimental design

- Experiment type: TMT
- PTM types: acetyl, phospho, redox
- Biological groups and comparisons: Mock vs Infected at 8h, 16h, and 24h; infected 8h vs 16h, 8h vs 24h, and 16h vs 24h
- Replicates: 4 per non-pooled group; pooled Total channels have 2 samples per group
- Batches: 1 and 2
- Pooled channels: `Total`
- Excluded samples and reasons: None

## Inputs

- Data directory: `scratch/ptm-demo/PTM`
- Global protein table: `scratch/ptm-demo/PTM/global_prot.tsv`
- Global peptide table: `scratch/ptm-demo/PTM/global_pept.tsv`
- PTM peptide tables: `acetyl_pept.tsv`, `phospho_pept.tsv`, `redox_pept.tsv`
- Reference FASTA: `scratch/ptm-demo/PTM/reference_proteome.fasta`
- Metadata table: `scratch/ptm-demo/PTM/metadata.tsv`
- Search/export preprocessing or column mapping: Standardized tab-separated demo inputs; no conversion performed

## Configuration decisions

- Log2 scale input: Yes
- Abundance correction: Enabled
- Paired abundance correction: Enabled
- Batch correction and samples: Enabled for configured TMT samples
- Missingness and replicate thresholds: `min_replicates_qc = 2`; `missing_thr = 1`
- Significance type and threshold: `pval`, `sig_thr = 0.05`
- FASTA ID matching and iBAQ: `fasta_id_matching = contains`; iBAQ enabled
- Other non-default settings: PTM symbols `@`, `#`, `@`; abbreviations `Ac`, `Ph`, `Ox`

## Execution

- Command or Python entry point: Inline Python using `.venv/bin/python`
- Analysis function: `ptm_analysis`
- Completion status: Passed
- Warnings or errors: None reported by the pipeline
- PTM output shape: 42,211 rows x 68 columns
- Global protein output shape: 5,817 rows x 66 columns
- Uncorrected output shape, if produced: Not produced

## Outputs

- Processed PTM site table: `scratch/ptm-demo/results/ptm_processed_site.csv`
- Processed global protein table: `scratch/ptm-demo/results/ptm_processed_prot.csv`
- Uncorrected PTM site table: Not produced
- Figures: None yet
- Derived tables: None yet

## QC and post-analysis

- Missingness findings: PTM output contains 245,610 missing cells, with sample-level missingness ranging from 13.7% to 17.7%. Global protein output contains 4,526 missing cells, with sample-level missingness ranging from 2.3% to 2.6%.
- Correlation/PCA findings: Core correlation/PCA figure generated; visual interpretation pending user review.
- PTM type counts: Ox 29,432; Ph 8,407; Ac 4,372.
- Significant-result filters: Core p-value and adjusted-p-value counts summarized at threshold 0.05; no rows filtered for final reporting.
- FDR recalculation: Global and protein-wise site FDR tables generated for all six comparisons.
- iBAQ or FASTA matching: All 5,817 global protein IDs matched the FASTA using configured `contains` mode; iBAQ values are stored in the original sample columns by the current implementation.
- Sequence, barcode, or coverage analysis: PTM-appropriate phosphopeptide coverage generated for protein `A0FGR8`. The barcode helper was not applicable because it expects LiP peptide fields such as `pept_type`, `pept_start`, and `pept_end`.
- Enrichment analysis and gene-set source/version: `gseapy` installed in `.venv`; enrichment not run because no GMT gene-set file is available.

QC artifacts:

- `scratch/ptm-demo/results/ptm_qc_sample_summary.csv`
- `scratch/ptm-demo/results/ptm_qc_type_counts.csv`
- `scratch/ptm-demo/results/ptm_qc_statistics_summary.csv`
- `scratch/ptm-demo/results/ptm_qc_samples.png`
- `scratch/ptm-demo/results/ptm_site_fdr_global.csv`
- `scratch/ptm-demo/results/ptm_site_fdr_proteinwise.csv`
- `scratch/ptm-demo/results/ptm_volcano_all_comparisons.png`
- `scratch/ptm-demo/results/ptm_phospho_coverage_A0FGR8.png`

## Interpretation and limitations

- Main observations: The core PTM processing and statistical pipeline completed successfully.
- Abundance-correction interpretation: Results represent PTM values corrected using global protein abundance under the paired-sample setting.
- Statistical or power limitations: Post-analysis review is pending; pooled channels were not part of the six configured comparisons.
- Data-quality limitations: Missingness and replicate-level QC have not yet been visualized or summarized by group.
- Biological caveats: No biological interpretation has been made at this draft stage.
- Remaining follow-up: Supply a versioned GMT gene-set file and rerun the approved enrichment step if enrichment is required.

## References

- [ProteoMeter documentation](../../../docs/index.rst)
- Focused API/source files reviewed: `src/proteometer/params.py`, `src/proteometer/ptm_analysis.py`, `src/proteometer/parse_metadata.py`, `src/proteometer/stats.py`, `src/proteometer/quality_control_plots.py`
