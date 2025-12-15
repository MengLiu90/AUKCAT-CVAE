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
The CVAE model uses substrate–EC number–species triples paired with experimentally measured kcat values as input instances. Substrates are embedded via [Mol2Vec](https://github.com/samoturk/mol2vec), EC numbers via [EC2Vec](https://github.com/MengLiu90/EC2Vec), and species via Node2Vec. 

The file ```./Datasets/Original_data.csv``` contains all input instances used to train the CVAE model described in the paper. 
The file ```./Datasets/example_data.csv``` contains example data after embedding, which is used as input to the CVAE model. 

## Train the CVAE Model
To train the CVAE model for synthetic data generation, simply run ```CVAE.py```.
The trained model will be saved to the ```./Trained_model``` directory.

## Generate Synthetic Data Using the Trained CVAE Model
To generate synthetic data points using the trained CVAE model, simply run ```CVAE_data_generation.py```. 
The number of synthetic instances per original data point can be configured via the ```n_replicates``` parameter. The generated dataset will be saved in the ```./Synthetic_data``` folder.
