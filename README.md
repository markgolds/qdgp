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
3. Install for local development:
   ```
   python -m poetry install
   ```

## Usage

Use notebook `run_models.ipynb` to experiment with models and datasets.

Or use the cluster with 1 disease per array:

1. Set up models in `run_parallel.py`

2. Submit a job to run; see `job_p.sh`

3. Once all jobs are complete, run `process_parallel.py` to combine all outputs and build plots.

4. Optionally run `plot_grid.py` to build a single grid of all resulting plots.
