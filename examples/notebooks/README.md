# Synthetic Notebook Corpus

Every notebook under this directory is synthetic. These are not human-created research
output and should not be used for scientific inference — they exist so the demo has
something realistic to index and answer questions about.

## Layout

Notebooks are a **flat list** at `examples/notebooks/*.ipynb`, except when a notebook
reads sibling files:

| Subfolder | Contents |
|-----------|----------|
| `synthetic_forecast/` | `synthetic_forecast.ipynb`, `weekly_sales.csv` |
| `synthetic_rotating_equipment_vibration/` | vibration notebook, `sensor_readings_SYN-442.csv` |
| `synthetic_mfg_yield/` | ingest / SPC / report notebooks, shared yield CSV |

Standalone engineering notebooks (`synthetic_chemical_reactor_calibration.ipynb`,
`synthetic_materials_tensile_test.ipynb`) live at the top level.

## Bioinformatics notebooks (flat)

- `synthetic_scrna_qc_clusters.ipynb` — single-cell RNA-seq QC and clustering
- `synthetic_bulk_rnaseq_hypoxia.ipynb` — bulk RNA-seq under hypoxia
- `synthetic_crispr_resistance_screen.ipynb` — CRISPR resistance screen
- `synthetic_rare_disease_variant_prioritization.ipynb` — rare disease trio
- `synthetic_microbiome_antibiotic_diversity.ipynb` — microbiome antibiotic exposure
- `synthetic_chipseq_enhancer_activation.ipynb` — ChIP-seq enhancer activation
- `synthetic_proteomics_kinase_inhibitor.ipynb` — phosphoproteomics kinase inhibitor
- `synthetic_spatial_tumor_neighborhoods.ipynb` — spatial tumor neighborhoods
- `synthetic_viral_lineage_amplicon.ipynb` — viral lineage amplicon QC
- `synthetic_multiomics_response_biomarkers.ipynb` — multi-omics response biomarkers

`nbrag-embed examples/notebooks` discovers every `.ipynb` here recursively.
