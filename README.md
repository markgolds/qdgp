# QDGP - Quantum Disease Gene Prioritisation

This repository contains the data and code used in the paper [Disease Gene Prioritization With Quantum Walks](https://arxiv.org/abs/2311.05486). 

## Installation
0. Install [Miniconda](https://docs.anaconda.com/free/miniconda/) if it's not already installed.

1. Clone this repository:
   ```
   git clone https://github.com/MarkEGold/qdgp
   ```
2. Create a Conda environment from the `qdgp.yaml` file in the root of the repository:
   ```
   conda env create -f qdgp.yaml
   ```
   Note that this environment is called `qdgp`. The new environment will
   contain Poetry.

3. Activate the new environment:
   ```
   conda activate qdgp
   ```
4. Use poetry to install required packages:
   ```
   python -m poetry install
   ```

## Usage

The two main programs of interest are 

- `cross-validate.py`, for doing cross-validation accross several models, as in the paper, and
	
- `predict.py`, for making predictions using the quantum walk method described in the corresponding paper.

### Cross-validation

Results from the paper can be reproduced by running `cross_validate.py` with the appropriate arguments. For example,

```
python cross_validate.py -n biogrid -d dgn --split_ratio 0.5 -runs 10
```

will run the cross-validation on the BioGRID PPI network with the DisGeNET data set using a train/test split of 50/50, with results being averaged over 10 runs. This will produce `out/dgn-biogrid-0.500.csv`, which can be used for further analysis, as well as plots in the `plots` directory.

### Predictions

Predictions can be made for any of the networks 

```
{"gmb", "wl", "biogrid", "string", "iid", "apid", "hprd"}
```

and any of the disease sets 

```
{"gmb", "dgn", "ot"}.
```

For a given a disease in the chosen disease set, the top N predictions of the quantum walk method can be calculated using `predictions.py`. For example:

```
python predictions.py --disease_set gmb --network wl --disease asthma --topn 200
```

will produce a `csv` file containing the top 200 predictions for the `asthma` seed genes contained in the `gmb` dataset, with the quantum walk being performed on the `wl` PPI network.

For a list of available diseases for a particular PPI network and disease set, you can run:

```
python avail_diseases.py -n biogrid -D dgn
```

as an example.

The time and diagonal hyperparameters can be set by modifying `predictions.py` accordingly.

Alternatively, custom disease datasets and/or PPI networks can be used by modifying the code in `qdgp/data.py`.
