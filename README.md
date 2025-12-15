# AUKCAT-CVAE
AUKCAT-CVAE is a conditional variational autoencoder designed to generate high-quality synthetic data to enhance training and generalization of AI models for kcat prediction.
Its underlying algorithm, however, is domain-agnostic and can be applied to any scenario requiring data augmentation for model generalization.
## Dependencies
1. pytorch 1.10.0
2. pandas 1.4.2
3. numpy 1.24.3
4. sklearn 1.0.2
5. scipy 1.5.3
6. matplotlib 3.7.2
7. CUDA 11.1

## Input data to the model
The CVAE model uses substrate–EC number–species triples paired with experimentally measured kcat values as input instances. Substrates are embedded via Mol2Vec, EC numbers via EC2Vec, and species via Node2Vec. 

The file ```./Datasets/Original_data.csv``` contains all input instances used to train the CVAE model described in the paper.

## Train the CVAE Model for Synthetic Instance Generation
To get trained CAVE model for synthetic data generation, simply run ```CVAE.py```, the trained model will be saved in ...
