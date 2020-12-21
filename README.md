Supplementary repository for the research on **An End-to-End Graph Neural Network for Disease Gene Prioritization**

This repository includes
* All code needed to reproduce the experiments.
* Instructions to setup the used environment to run the experiments.
* The datasources needed as Inputs for the experiments.
* The evaluation results.
* The pre trained models.

The process of the experiments is ducomented in
* [experiments/disease_gene_prioritization.ipynb for the disease gene prioritization task.](experiments/disease_gene_prioritization.ipynb)
* [experiments/disease_gene_classification.ipynb for the disease type classification.](experiments/disease_gene_classification.ipynb)

# Setup
## Preliminaries
The experiments have been performed using `python 3.7` and this hardware:
* AMD Ryzen 7 2700X Eight-Core Processor
* 34 GB RAM
* GeForce RTX 2080 Ti

### Setup python environment.
## Using Conda
```bash
conda create --name dpg_gnn python=3.7
conda install -y -q --name dpg_gnn -c conda-forge --file requirements.txt
conda activate dpg_gnn
```

## Using Virtualenv + pip
```bash
virtualenv dpg_gnn -p `which python3.7`
source dpg_gnn/bin/activate
pip install -r requirements.txt
```

# Data sources
### [HumanNet-XN.tsv](data_sources/HumanNet-XN.tsv)
**Description:**<br>
HumanNet v2: human gene networks for disease research.

**Source:**<br>
[inetbio.org/humannet](https://www.inetbio.org/humannet/download.php)

### [all_diseases.tsv](data_sources/all_diseases.tsv)
**Description:**<br>
Names and OMIM Ids of all diseases covered in this experiment.

**Source:**<br>
* [Online Mendelian Inheritance in Man (OMIM)]()
* [Mouse alleles from Mouse Genome Informatics (MGI)](http://www.informatics.jax.org/)

### [disease_hpo.tsv](data_sources/disease_hpo.tsv)
**Description:**<br>
Human Phenotype Ontology Annotations associated to diseases via OMIM ids.

**Source:**<br>
[Human Phenotype Ontology](https://hpo.jax.org/app/download/annotation)

### [disease_publication_titles_and_abstracts.tsv](data_sources/disease_publication_titles_and_abstracts.tsv)
**Description:**<br>
Titles and abstracts of publications associated to diseases identified via OMIM ids.

**Source:**<br>
[NCBI pubmed](ftp://ftp.ncbi.nih.gov/gene/DATA/gene2pubmed.gz)

### [gene_expressions.tsv](data_sources/gene_expressions.tsv)
**Description:**<br>
Gene expression conditions extracted from human gene expression atlas of 5372 samples representing 369 different cell
 and tissue types, disease states and cell lines.

**Source:**<br>
https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-62/

### [gene_hpo_disease.tsv](data_sources/gene_hpo_disease.tsv)
**Description:**<br>
Human Phenotype Ontology Annotations associated to genes via NCBI entrez gene ids.
 	
**Source:**<br>
[Human Phenotype Ontology](https://hpo.jax.org/app/download/annotation)


### [extracted_disease_class_assignments](data_sources/extracted_disease_class_assignments.tsv)
**Description:**<br>
Gene disease class associations as created by the Human Disease Network

**source**<br>
* [The human disease network](https://doi.org/10.1073/pnas.0701361104)
* [. Curated Morbid Map file with disease ID and class assignment (December 21, 2005 version).](https://www.pnas.org/content/pnas/suppl/2007/05/03/0701361104.DC1/01361Table1.pdf)

### [gene_ontologies.tsv](data_sources/gene_ontologies.tsv)
**Description:**<br>
Gene ontology terms associated to genes by entrez gene ids.

Example:
```text
ID  Parent      Evidence code   GO term
18  GO:0001666  IEA             GO:0009628
```

Means Gene 18 has term GO:0009628 where GO:0001666 is the parent end the
[GO evidence code](http://geneontology.org/docs/guide-go-evidence-codes/) is `IEA` (Inferred from Electronic
 Annotation).

**Source:**<br>
[The Gene Ontology Resource](http://geneontology.org/)

### [genes_diseases.tsv](data_sources/genes_diseases.tsv)
**Description:**<br>
Genes associated to diseases (Entrez gene id, OMIM disease id).

**Source:**<br>
* [Online Mendelian Inheritance in Man (OMIM)](https://www.omim.org/)
* [Mouse Genome Informatics (MGI)](http://www.informatics.jax.org/)

### [genes_diseases.tsv](data_sources/genes_diseases_mgi_only.tsv)
**Description:**<br>
Genes associated to diseases (Entrez gene id, OMIM disease id).

**Source:**<br>
* [Mouse Genome Informatics (MGI)](http://www.informatics.jax.org/)

### [CTD_chemicals_diseases.tsv.gz](data_sources/CTD_chemicals_diseases.tsv.gz)
**Source:**<br>
* http://ctdbase.org/downloads/

# Content
```
.
|-- DiseaseNet.py
|-- GeneNet.py
|-- README.md
|-- TheModel.py
|-- data_sources
|   |-- CTD_chemicals_diseases.tsv.gz
|   |-- HumanNet-FN.tsv
|   |-- HumanNet-XN.tsv
|   |-- all_diseases.tsv
|   |-- disease_hpo.tsv
|   |-- disease_net_pubmed_knn
|   |   |-- processed
|   |   |   |-- data.pt
|   |   |   |-- disease_id_feature_index_mapping.txt
|   |   |   |-- edges.pt
|   |   |   |-- nodes.pt
|   |   |   |-- pre_filter.pt
|   |   |   `-- pre_transform.pt
|   |   `-- raw
|   |       |-- CTD_chemicals_diseases.tsv.gz
|   |       |-- all_diseases.tsv
|   |       |-- disease_hpo.tsv
|   |       |-- disease_pathway.tsv
|   |       `-- disease_publication_titles_and_abstracts.tsv
|   |-- disease_pathway.tsv
|   |-- disease_publication_titles_and_abstracts.tsv
|   |-- extracted_disease_class_assignments.tsv
|   |-- gene_expressions.tsv
|   |-- gene_gtex_rna_seq_expressions.tsv
|   |-- gene_hpo_disease.tsv
|   |-- gene_net_fn_hpo
|   |   |-- processed
|   |   |   |-- data.pt
|   |   |   |-- edges.pt
|   |   |   |-- gene_id_data_index.tsv
|   |   |   |-- nodes.pt
|   |   |   |-- pre_filter.pt
|   |   |   `-- pre_transform.pt
|   |   `-- raw
|   |       |-- HumanNet-FN.tsv
|   |       |-- gene_expressions.tsv
|   |       |-- gene_gtex_rna_seq_expressions.tsv
|   |       |-- gene_hpo_disease.tsv
|   |       |-- gene_ontologies.tsv
|   |       `-- gene_pathway_associations.tsv
|   |-- gene_ontologies.tsv
|   |-- gene_pathway_associations.tsv
|   |-- genes_diseases.tsv
|   `-- genes_diseases_mgi_only.tsv
|-- experiments
|   |-- disease_gene_classification.ipynb
|   |-- disease_gene_prioritization.ipynb
|   `-- results
|       `-- final
|           |-- Disease_gene_prediction_ROC_by_fold_monogenic_diseases.pdf
|           |-- Disease_gene_prediction_ROC_by_fold_multigenic_diseases.pdf
|           |-- Disease_gene_prediction_ROC_combined.pdf
|           |-- disease_classification_results.gz
|           |-- disease_gene_classification_result_bar_chart_Pr-auc.pdf
|           |-- disease_gene_classification_result_bar_chart_ROC-auc.pdf
|           |-- disease_gene_classification_result_bar_chart_ROC-auc_Pr-auc_fmax.pdf
|           |-- disease_gene_classification_result_bar_chart_fmax.pdf
|           |-- final_hyperparameters_dis_dict.pickle.gz
|           |-- final_hyperparameters_metrics.pickle.gz
|           |-- model_fold_1.ptm
|           |-- model_fold_2.ptm
|           |-- model_fold_3.ptm
|           |-- model_fold_4.ptm
|           `-- model_fold_5.ptm
`-- requirements.txt
```
