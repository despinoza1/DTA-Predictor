# DTA-Predictor

My Senior Project for my Bachelor's degree in Computer Science for predicting the binding affinity of drug compounds and their targets. 
The binding affinity is predicted as a pKd value; `pKd = -log(Kd)`.  Deep Convolutional Neural Network implementation was inspired by [AtomNet](https://arxiv.org/abs/1510.02855v1) and [Pafnuncy](https://arxiv.org/abs/1712.07042v2).

Advisor: [Dr. Dongchul Kim](https://faculty.utrgv.edu/dongchul.kim/)

## Data

Used the Kd values from interactions in the [Drug Target Commons](https://drugtargetcommons.fimm.fi/).
3D structure of compounds and proteins were obtained from [ChemBL](https://www.ebi.ac.uk/chembl/) and [UniProt](https://www.uniprot.org/) respectively.

## Usage

### Prepare

`python prepare.py <input file> --output <output complexes> --path <path to data>`

The input file is a CSV file with names of compound and target 
The path is to the location of the 3D structure data of each compound and target 
  
### Predict

`python predict.py <input complexes> --output <output file> --csv-file <csv file> --model <model file>`

Input complexes is the complex of a drug compound and its target
CSV file is the file with the drug compound ID and Target ID which a new column with their predicted binding affinity will be added
Model file is the model to use for prediction

### Train

`python train.py <input complexes> --output <output file>`

Parameters that can be changed:

- Learning Rate: `--learning-rate <float>`
- Batch Size: `--batch-size <int>`
- Percentage used for validation: `--percent <float>`
- Epochs: `--epochs <int>`
- Dropout Rate: `--dropout <float>`

## Dependencies

The Python dependencies can be installed by using `pip install -r requirements.txt`
