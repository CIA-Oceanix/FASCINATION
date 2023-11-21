# FASCINATION

## Repository for the FASCINATION project

This repository contains the code for the part of the **FASCINATION** project (FAisabilité d’un Système Cognitif à INtelligence Artificielle pour le Traitement de l’Information Océanographique Nomade) dedicated to optimizing and compressing the sound speed information. This compression is conducted through the use of a AutoEncoder neural network, whose objective is to compress Sound Speed Fields (SSF) into a smaller vector that contains most of the information of the initial data. The code for the AutoEncoder can be found in `src/autoencoder.py`. 

It is guided in its compression by another neural network which can be found in `src/acoustic_predictor.py`. The role of this network is to constrain the compression in order to make sure that the essential information in regard to specific acoustic variables is preserved. In other words, during the training of the AutoEncoder, it takes as input the output of the AutoEncoder (ie the decompressed SSF), and predicts specific acoustic variables of our choosing. The loss associated to the prediction of the acoustic variables is then taken into account for the training of the AutoEncoder. This helps making sure that the information related to certain acoustic variables is better preserved during the compression than some other information which might be less useful. So far, the variables it predicts are the ECS and the cutoff frequency associated to it.

## How to use the project

### Install
---
#### Install project dependencies
```
https://github.com/CIA-Oceanix/FASCINATION.git
cd FASCINATION
pip install -r requirements.txt
```
### Run
---
The model uses [Hydra](https://hydra.cc/). Assuming you downloaded the project's data from a reliable source, you can run the training of the acoustic predictor using:

```
python main.py xp=acoustic_pred_training paths.sound="your path to the sound file" paths.variables="your path to the acoustic variables file"
```
### Useful links
- [Hydra documentation](https://hydra.cc/)
- [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/)