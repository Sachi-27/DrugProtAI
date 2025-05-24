
## DrugProtAI

The implementation of our paper [DrugProtAI : A guide to the future research of investigational target proteins](https://doi.org/10.1101/2024.11.05.622045). For questions, feedback, or support, please reach out to us at [bddfresearch@gmail.com](mailto:bddfresearch@gmail.com).

**DrugProtAI** is a machine learning platform designed to predict the druggability of human proteins, leveraging advanced algorithms, protein embeddings, and comprehensive biological data. The [tool](https://drugprotai.pythonanywhere.com/) aims to assist researchers in drug discovery by identifying and prioritizing proteins with the potential to be drug targets.

## 📂 Data

- The platform uses a manually curated dataset containing 20,273 human proteins, cross-referenced with DrugBank v5.1.12(March 14, 2024), categorizing proteins into druggable, investigational, and non-druggable classes. 
- Protein embeddings and sequence information are sourced from UniProt
- Second cross-referencing with Drugbank v5.1.13(January 02, 2025) observes 81 additional proteins with approved drugs

The entire feature set (183-dim) is available at the following location:

🔗 [Download dataset](https://drive.google.com/file/d/1iA2n7_Mp6obDSEmrmqJ8dujS26wW97ce/view?usp=sharing)

## 🌟 Key Features

- **Druggability Index**: It quantifies the likelihood of a protein being druggable.
- **Comprehensive Feature Set**: Incorporates 183 features derived from SwissProt-verified data, including domains, protein-protein interactions (PPI), subcellular locations, post translational modifications, physicochemical properties and sequence derived features.
- **Partition Ensemble Classifier (PEC)**: Employs a partitioned ensemble of models trained on balanced subsets of the dataset to address class imbalance and improve prediction accuracy.
- **Partition Leave-One-Out Ensemble Classifier (PLOEC)**: This is used for computing DI for a non-druggable protein(which is in the training set), it evaluates in a similar fashion to PEC, except that it excludes the model which is trained on the particular partition containing the queried protein.
- **Explainability with SHAP**: Provides SHAP values to explain the contributions of different features to the model’s predictions.
- **Blind Validation**: Evaluates druggability index using PEC on investigational proteins and PLOEC on non-druggable proteins. The predictions are compared against the new proteins with approved drugs. Evaluation metric is accuracy

## 🌲Citation
If you find our work useful, feel free to cite our paper:
```
@article {HalderDrugProtAI,
	author = {Halder, Ankit and Samantaray, Sabyasachi and Barbade, Sahil and Gupta, Aditya and Srivastava, Sanjeeva},
	title = {DrugProtAI: A guide to the future research of investigational target proteins},
	elocation-id = {2024.11.05.622045},
	year = {2024},
	doi = {10.1101/2024.11.05.622045},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Drug design and development are central to clinical research, yet ninety percent of drugs fail to reach the clinic, often due to inappropriate selection of drug targets. Conventional methods for target identification lack precision and sensitivity. While various computational tools have been developed to predict the druggability of proteins, they often focus on limited subsets of the human proteome or rely solely on amino acid properties. To address the challenge of class imbalance between proteins with and without approved drugs, we propose a novel Partitioning Method. We evaluated the druggability potential of 20,273 reviewed human proteins, of which 2,636 have approved drugs. Our comprehensive analysis of 183 features, encompassing biophysical and sequence-derived properties, achieved a median AUC of 0.86 in target predictions. We utilize SHAP (Shapley Additive Explanations) scores to identify key predictors and interpret their contribution to druggability. We have reviewed and evaluated 688 investigational proteins from DrugBank (https://go.drugbank.com/) using our tool, DrugProtAI (https://drugprotai.pythonanywhere.com/). Our tool offers druggability predictions and access to 2M+ publications on drug targets and their effects, aiding in the selection of target proteins for drug development. We believe that insights into key predictors will significantly advance drug development and propel the field forward.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/11/08/2024.11.05.622045},
	eprint = {https://www.biorxiv.org/content/early/2024/11/08/2024.11.05.622045.full.pdf},
	journal = {bioRxiv}
}
```


