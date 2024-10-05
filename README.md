
# DrugProtAI

**DrugProtAI** is a machine learning platform designed to predict the druggability of human proteins, leveraging advanced algorithms, protein embeddings, and comprehensive biological data. The [tool](https://drugprotai.pythonanywhere.com/) aims to assist researchers in drug discovery by identifying and prioritizing proteins with the potential to be drug targets.

## Key Features

- **Druggability Index**: Utilizes a novel Druggability Index (DI) metric, which quantifies the likelihood of a protein being druggable based on its biological characteristics.
- **Comprehensive Feature Set**: Incorporates 183 features derived from SwissProt-verified data, including domains, protein-protein interactions (PPI), subcellular locations, post translational modifications, physicochemical properties and sequence derived features.
- **Partition Ensemble Classifier (PEC)**: Employs a partitioned ensemble of models trained on balanced subsets of the dataset to address class imbalance and improve prediction accuracy.
- **Partition Leave-One-Out Ensemble Classifier (PLOEC)**: This is used for computing DI for a non-druggable protein(which is in the training set), it evaluates in a similar fashion to PEC, except that it excludes the model which is trained on the particular partition containing the queried protein.
- **Explainability with SHAP**: Provides SHAP values to explain the contributions of different features to the model’s predictions.
- **Blind Validation**: Evaluates druggability index using PEC on investigational proteins and PLOEC on non-druggable proteins. The predictions are compared against the new proteins with approved drugs. Evaluation metric is accuracy

## Data

- The platform uses a manually curated dataset containing 20,273 human proteins, cross-referenced with DrugBank (March 16, 2024), categorizing proteins into druggable, investigational, and non-druggable classes. 
- Protein embeddings and sequence information are sourced from UniProt
- Second cross-referencing with Drugbank (August 10, 2024) observes 33 additional proteins with approved drugs

## Web Application

DrugProtAI is hosted as a web application. Key functionalities include:
1. **Protein Search**: Retrieve protein-specific information from UniProt and DrugBank.
2. **Feature Contribution**: Visualize top-K feature contributions to druggability predictions.
3. **3D Protein Viewer**: View protein structures using AlphaFold.
4. **Druggability Index**: Display DI scores for investigational and non-druggable proteins.
5. **Drug Information**: Retrieve and display publications which evidence drugs associated with proteins.
