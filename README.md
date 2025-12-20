# ProteoMeter


<p align="center">
<a href="https://pypi.python.org/pypi/proteometer">
    <img src="https://img.shields.io/pypi/v/proteometer.svg"
        alt = "Release Status">
</a>
<!-- 
<a href="https://github.com/PhenoMeters/proteometer/actions">
    <img src="https://github.com/PhenoMeters/proteometer/actions/workflows/main.yml/badge.svg?branch=release" alt="CI Status">
</a> -->
<a href="https://PhenoMeters.github.io/proteometer/">
    <img src="https://img.shields.io/website/https/PhenoMeters.github.io/proteometer/index.html.svg?label=docs&down_message=unavailable&up_message=available" alt="Documentation Status">
</a>

</p>


## ProteoMeter: A Comprehensive Python Library for Proteomic Data Analysis

* Installation: `pip install proteometer`
* Free software: BSD 2-Clause License [(see LICENSE)](https://github.com/PNNL-Predictive-Phenomics/ProteoMeter/blob/main/LICENSE)
* Documentation and API Reference: <https://pnnl-predictive-phenomics.github.io/ProteoMeter/>


## Description:

ProteoMeter is an innovative Python library designed to revolutionize the way researchers approach proteomic data. This project aims to provide a robust and user-friendly toolkit for processing, integrating, and analyzing proteomic data, leveraging unified structural coordinates and standardized methods.

## Key Features:

Data Processing Capabilities: ProteoMeter offers advanced functionalities for preprocessing and cleaning proteomic datasets, ensuring data quality and consistency.

Integration Tools: With its ability to seamlessly integrate diverse proteomic data sources, ProteoMeter enables researchers to combine datasets from different experiments or platforms, enhancing the depth and breadth of analysis.

Unified Structural Coordinates: Utilizing a unified coordinate system, ProteoMeter facilitates the accurate comparison and overlay of protein structures, making it easier to identify structural similarities and differences.

Comprehensive Analysis Suite: The library includes a wide array of analytical tools, from basic protein quantification to advanced computational methods for protein interaction mapping, post-translational modification analysis, and functional annotation.

User-Friendly Interface: Designed with the end-user in mind, ProteoMeter offers an intuitive interface that caters to both novice users and experienced bioinformaticians.

Extensible and Modular Design: The modular nature of the library allows for easy expansion and customization, ensuring that ProteoMeter remains at the forefront of proteomic research developments.

## Use Cases:

Academic research in proteomics, molecular biology, and biochemistry.
Pharmaceutical and biotech industries for drug discovery and protein analysis.
Clinical research for biomarker discovery and disease profiling.
ProteoMeter is not just a tool but a stepping stone towards a more integrated and comprehensive understanding of proteomic data, aiming to accelerate research and discovery in the field of protein science.

## Required Inputs:
Example data and configuration files are in [demo_data](https://github.com/PNNL-Predictive-Phenomics/ProteoMeter/tree/main/demo_data). The file names corresponding to each input are configurable; below we use the default names.

#### Configuration and Metadata Files
* metadata.tsv
    * Defines the group ID, replicate number, condition (treatment vs control), treatment type, and other variables (e.g., sample time or strain).
* config.toml (lip.toml and/or ptm.toml in the demo_data example)
    * Specifies experiment conditions that affect processing and determines which statistics to compute. See the commented example configs in demo_data for detailed information.
* reference_proteom.fasta
    * Fasta file containing sequence information for the proteome.

#### LiP Data
* lip_pept.tsv: quantification of peptides after limited proteolysis followed by trypsin digestion (ProK + Trypsin)
* trypsin_pept.tsv: quantification of peptides after trypsin digestion without limited proteolysis (Trypsin only)
* trypsin_prot.tsv: quantification of peptides after trypsin digestion without limited proteolysis (Trypsin only)

In the demo_data/LiP directory, we have also included lip_prot.tsv, which contains quantification of proteins after limited proteolysis followed by trypsin digestion. This file is not required, but may be used in place of trypsin_prot.tsv (generally this is not recommended as the abundance quantification will likely be more robust for trypsin-digestion than for limited proteolysis followed by trypsin digestion).

#### PTM Data
* ptm_pept.tsv: quantification of peptides for a given PTM type
* ptm_prot.tsv: quantification of proteins derived from the peptide-level data

The files above correspond to a specific ptm type (e.g., LYS acetylation). Multiple such files can be provided in a single run. In the demo_data example, we simultaneously analyze LYS acentylaion (acetyl_pept.tsv and acetyl_prot.tsv), CYS oxidation (redox_pept.tsv and redox_prot.tsv), and SER/THR phosphorylation (phospho_pept.tsv and phospho_prot.tsv).

## Outputs:
`ProteoMeter` generates `pandas` DataFrame objects. In our demonstration, we have saved these outputs to CSV files, each of which corresponds to a data frame generated by the `ProteoMeter` `lip_analysis` or `ptm_analysis` functions. In all outputs, fold changes are given in a base 2 logarithmic scale.
* lip_processed_pept.csv: LiP double-digested (ProK + Trypsin) peptide-level data. This includes fold-changes for all peptides, tryptic or otherwise (tryptic peptides are indicated by the pept_type column.)
* lip_processed_prot.csv: Global abundance information used for abundance correction of peptide changes. This includes abundance quantification, fold-changes and statistical significance.
* lip_processed_site.csv: Digestion cut site quantification, fold-changes and statistical significance. 
* ptm_processed_site.csv: PTM site quantification, fold-changes and statistical significance.
* ptm_processed_prot.csv: Global abundance information used for abundance correction of PTM changes. This includes abundance quantification, fold-changes and statistical significance.