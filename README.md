# QDGP - Quantum Disease Gene Prioritisation

## Installation
1. Create a Conda environment from the `qdgp.yaml` file in the root of the repository:
   ```
   conda env create -f qdgp.yaml
   ```
   Note that this environment is called `qdgp`. The new environment will
   contain Poetry.

2. Activate the new environment:
   ```
   conda activate qdgp
   ```
3. Use poetry to install required packages:
   ```
   python -m poetry install
   ```

## Usage

The two main programs of interest are 

	- `predict.py`, for making predictions using the quantum walk method described in[PAPER], and 
	
	- `cross-validate.py`, for doing cross-validation accross several models, as in the paper.
	
### Predictions

Predictions can be made for any of the networks `{"gmb", "wl", "biogrid", "string", "iid", "apid", "hprd"}` and any of the disease sets `{"gmb", "dgn", "ot"}`. (See paper for descriptions). For a given a disease in the chosen disease, the top N predictions of the quantum walk method can be calculated using `predictions.py`. For example:
```
python predictions.py --disease_set gmb --network wl --disease asthma --topn 200
```
will produce a `csv` file containing the top 200 predictions for the `asthma` seed genes contained in the `gmb` dataset, with the quantum walk being performed on the `wl` PPI network.

Alternatively, custom disease datasets and/or PPI networks can be used by modifying the code in `qdgp/data.py`.

### Cross-Validation
