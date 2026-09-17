---
name: proteometer-ptm-workflow
description: "Guide typical ProteoMeter PTM processing and analysis workflows: organize standardized modified-peptide, global peptide, global protein, and FASTA inputs; create and validate metadata.tsv and ptm.toml; run Params and ptm_analysis; diagnose schema/configuration failures; and perform QC, statistics, visualization, FASTA/iBAQ, and enrichment post-analysis. Use when a user asks to configure, run, troubleshoot, or interpret a ProteoMeter PTM analysis."
compatibility: Requires the ProteoMeter repository, Python 3.12+, and its configured environment; enrichment examples additionally require the optional demo dependencies and a downloaded gene-set file.
metadata:
  author: ProteoMeter contributors
  version: "1.0"
---

# ProteoMeter PTM workflow

Use this skill for an end-to-end PTM proteomics analysis. Work from the user's real files and preserve their column names where possible; do not invent a new pipeline when the repository already provides an equivalent helper.

This skill is for the typical PTM workflow only. Do not extend its PTM input assumptions to LiP or other ProteoMeter workflows without inspecting their dedicated configs and APIs.

## Human approval gates

Treat every workflow stage below as a checkpoint. After completing a stage, stop and present a concise handoff containing:

- What was inspected, created, or changed.
- The files, sample counts, assumptions, and issues involved.
- The exact next action and any decision the user must make.

Do not begin the next stage until the user explicitly approves it. A response such as "approved", "continue", or a clear answer to the stated decision is required. If the user requests changes, revise the current stage and present that stage's handoff again.

## Source of truth

Before changing inputs or code, inspect these local examples and APIs:

- `demo_data/ptm.toml` for the complete PTM configuration shape.
- `demo_data/PTM/metadata.tsv` for metadata semantics and sample naming.
- `demonstration.py` for the minimal executable PTM run.
- `demonstration.ipynb` for QC, multiple-testing, plotting, FASTA, and enrichment follow-up.
- `src/proteometer/params.py` for config validation and resolved paths.
- `src/proteometer/ptm_analysis.py` for pipeline order and returned DataFrames.
- `src/proteometer/quality_control_plots.py` and `src/proteometer/stats.py` for post-analysis APIs.
- `../../../docs/index.rst` for the repository's full documentation entry point and `../../../src/proteometer/` for the complete API source when the focused reference is insufficient.

Read `references/ptm-reference.md` when you need exact input columns, configuration invariants, output interpretation, or failure diagnosis.
Use `assets/ptm-analysis-summary.md` as the durable report template.

## Workflow

Track these gates in order:

- [ ] Inventory the quantification tables, sample columns, PTM labels, metadata, and FASTA.
- [ ] Normalize the input layout and verify table schemas before writing configuration.
- [ ] Create `metadata.tsv` with sample IDs that exactly match every quantitative table.
- [ ] Create `ptm.toml`, keeping PTM names, files, symbols, and abbreviations positionally aligned.
- [ ] Load `Params` and run a preflight check before the full analysis.
- [ ] Run `ptm_analysis` or `ptm_analysis_return_all` and persist outputs.
- [ ] Perform QC and statistical review before biological interpretation.
- [ ] Apply optional downstream analyses such as protein-wise FDR, FASTA/iBAQ, sequence/barcode views, or enrichment.
- [ ] Report assumptions, exclusions, thresholds, output paths, and unresolved data-quality issues.

### 1. Inventory and map the data

Identify:

- One reference FASTA.
- Global protein and global peptide quantification tables. These provide abundance correction context.
- One peptide-level table per PTM type.
- The sample columns and whether intensities are already log2-scaled.
- The search tool and experiment type (`TMT` or `Label-free`).
- Replicates, groups, conditions, batches, pooled channels, and intended comparisons.

Do not assume that a vendor/search-engine export is directly compatible. Compare its headers and PTM notation with the demo and package parsing code first. Preserve raw inputs and write cleaned or renamed copies into a dedicated analysis data directory.

ProteoMeter's analysis functions expect standardized tab-separated quantification tables; they do not perform a general MaxQuant, FragPipe, or MSFragger export conversion for PTM input. If the user's files are raw search-engine exports, identify or perform that conversion before this workflow and document the mapping.

**Gate 1:** Report the discovered files, table roles, sample columns, PTM types, FASTA choice, and any required conversion or unresolved mismatch. Ask the user to approve the input mapping before creating metadata.

### 2. Build and validate metadata

Use one row per quantitative sample. The configured sample column must contain exact, case-sensitive column names from the global and PTM tables. Include the configured batch, group, and condition columns; add biological design columns such as replicate, time, dose, or treatment when they are useful as ANOVA factors or for auditability.

Distinguish group identity from condition role. In the demo, `Mock_8h` and `Infected_8h` are groups, while `Control` and `Treatment` identify condition roles. Define pairwise comparisons using group names, not condition labels, and ensure each comparison has enough replicates after any exclusions.

Check that every metadata sample is present in the quantitative tables and that every quantitative sample is represented in metadata. If this is false, stop and resolve the mapping instead of silently dropping columns.

**Gate 2:** Show the proposed metadata columns, sample-to-group/condition/batch mapping, replicate structure, and planned comparisons. Ask the user to approve the experimental design mapping before writing or editing `metadata.tsv`.

### 3. Create `ptm.toml`

Start from `demo_data/ptm.toml` and adapt paths, column names, PTM files, symbols, abbreviations, experiment settings, corrections, and comparisons. For PTM data, configure `paths.ptm.ptm_pept_files`; leave the LiP-only path empty unless the same config is intentionally shared with a LiP run.

Keep these lists in the same order:

- `symbols.ptm.ptm_names`
- `paths.ptm.ptm_pept_files`
- `symbols.ptm.ptm_symbols`
- `symbols.ptm.ptm_abbreviations`

Set `log2_scale` to reflect the actual input. Use abundance correction when the biological question calls for modified-state changes relative to protein abundance, and ensure global protein/peptide data are sample-paired when `abundance_correction_paired_samples = true`. Enable batch correction for an appropriate TMT design and provide the sample columns listed in `batch_correct_samples`.

Set `ibaq = true` only when the FASTA identifiers can be matched using the configured `fasta_id_matching` mode. Confirm `sig_type`, thresholds, and `min_replicates_qc` match the intended analysis rather than accepting demo defaults blindly.

**Gate 3:** Summarize the proposed TOML paths, PTM list alignment, corrections, thresholds, and exclusions. Ask the user to approve the configuration before running preflight.

### 4. Preflight before analysis

Run a small Python check that loads `Params`, reads all TSV files with `sep="\t"`, and verifies:

- All configured files exist.
- Required identifier columns exist in every applicable table.
- Metadata sample IDs and quantitative columns agree.
- PTM list lengths agree.
- PTM peptide tables contain the configured peptide, UniProt, protein, and residue columns.
- Residue strings and PTM-marked peptide sequences are parseable according to the package conventions.
- No group loses all or too many replicates after planned `drop_samples`.

Then instantiate `Params("path/to/ptm.toml")`. This catches invalid experiment type, significance type, search tool, and PTM list lengths before the expensive pipeline starts.

If `skills-ref` is installed, validate the skill itself from the repository root:

```bash
skills-ref validate .agents/skills/proteometer-ptm-workflow
```

When `skills-ref` is unavailable, perform this minimal manual fallback:

```bash
ruby -e 'require "yaml"; text = File.read(".agents/skills/proteometer-ptm-workflow/SKILL.md"); raise "missing frontmatter" unless text.start_with?("---\\n"); data = YAML.safe_load(text.split("---\\n", 3)[1]); raise "bad name" unless data["name"] == "proteometer-ptm-workflow"; raise "missing description" unless data["description"].is_a?(String) && !data["description"].empty?; puts "frontmatter: valid"'
test -f .agents/skills/proteometer-ptm-workflow/references/ptm-reference.md
```

Also confirm that the skill directory name matches the lowercase hyphenated `name`, that the frontmatter is delimited by `---`, and that referenced local files exist. This validates structure only; it does not replace an actual ProteoMeter preflight or analysis run.

**Gate 4:** Present the preflight results, including missing files or columns, sample mismatches, parse warnings, and replicate-count risks. Ask the user to approve the validated inputs and configuration before starting `ptm_analysis`.

### 5. Run and persist the pipeline

Use the package API rather than reproducing internal processing:

```python
from proteometer.params import Params
from proteometer.ptm_analysis import ptm_analysis

params = Params("path/to/ptm.toml")
ptm_site, global_prot = ptm_analysis(params, drop_samples=None)
ptm_site.to_csv("path/to/results/ptm_processed_site.csv")
global_prot.to_csv("path/to/results/ptm_processed_prot.csv")
```

Use `drop_samples=[...]` only when exclusions are documented and leave enough replicates per group. Use `ptm_analysis_return_all(params)` when the uncorrected site-level result is needed for comparison with abundance-corrected output.

Remember the returned order: `ptm_analysis` returns combined PTM/global site-level output first and processed global protein output second. The `return_all` variant adds uncorrected PTM output as the third result.

**Gate 5:** Report whether the analysis completed, output shapes, output paths, warnings, and the corrections actually applied. Ask the user to approve proceeding to QC and post-analysis.

At this point, create a draft Markdown summary by copying `assets/ptm-analysis-summary.md` into the configured results directory, for example `ptm_analysis_summary.md`. Populate the design, inputs, configuration, execution status, output shapes, output paths, warnings, and corrections applied. The draft records the run before optional downstream interpretation; update the same file after Gate 6 rather than creating a separate competing summary.

### 6. QC and post-analysis

Before interpreting biology, inspect row counts, PTM types, missingness, sample correlations, replicate separation, comparison columns, and the distribution of raw or adjusted p-values. Follow the notebook's established defaults where applicable:

- `quality_control_plots.correlation_plot` for sample-level correlation.
- `quality_control_plots.biplot` for PCA-style replicate and group separation.
- `quality_control_plots.volcano_plot` for comparison-level effect/significance review.
- `stats.recalculate_adj_pval` for global FDR recalculation and `stats.recalculate_adj_pval_proteinwise` for protein-wise FDR when that analysis is appropriate.
- FASTA helpers and `abundance.calculate_ibaq_from_fasta` for sequence matching or iBAQ checks.
- Barcode/alignment and peptide-coverage plots for site or sequence context.
- The notebook's `gseapy` example for enrichment only after selecting a defensible significant protein set and obtaining the required GMT file.

Use `parse_metadata.int_columns` and `parse_metadata.group_columns` with the loaded `Params` when constructing QC plots or checking sample groups; this avoids manually guessing which columns are quantitative or pooled.

Make comparison names from the actual `ttest_pairs` labels and inspect generated column names rather than assuming a fixed suffix. Treat missing p-values as a data-quality or power signal, not as evidence of no effect.

**Gate 6:** Present QC findings, significant-result filters, plots/tables created, and any proposed FDR, iBAQ, sequence, or enrichment follow-up. Ask the user which approved post-analysis actions to perform, and do not run optional downstream analyses until they choose.

After the approved post-analysis actions finish, update `ptm_analysis_summary.md` with QC results, selected filters, plots and derived tables, approved follow-up analyses, biological interpretation caveats, and remaining limitations. Present the updated file contents or a concise excerpt at the final approval gate.

## Failure handling

When a run fails, classify the failure before editing data:

1. Path/configuration failure: inspect `Params` and resolved absolute paths.
2. Schema/sample mismatch: compare metadata and table headers exactly.
3. Parsing failure: inspect peptide modification notation and residue formats.
4. Statistical/QC failure: inspect replicate counts, missingness, batches, and comparison group names.
5. FASTA/enrichment failure: verify identifier matching and optional external inputs.

Fix the smallest upstream cause, rerun the relevant preflight or analysis, and record the change. Never hide a mismatch by dropping unmatched samples without reporting it.

## Completion criteria

A workflow is complete only when the configuration loads, the analysis finishes, outputs are written, output types and comparison columns are inspected, QC has been run, and the final report states the design, corrections, exclusions, thresholds, and limitations. Cite the concrete input and output paths in the handoff.

Before finalizing, provide a final handoff for user approval that lists the completed stages, output files, analysis decisions, and remaining limitations. Ensure the finalized `ptm_analysis_summary.md` is saved beside the analysis outputs and links back to the full repository documentation when deeper API details are relevant. Do not describe the workflow as complete until the user approves the final handoff.
