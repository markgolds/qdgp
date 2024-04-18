"""Get the available diseases for a chosen PPI network and disease set."""

import argparse
from typing import Dict

import qdgp.data as dt


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description="List available diseases for a network and disease set.",
    )
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        default="gmb",
        choices=dt.valid_networks,
    )
    parser.add_argument(
        "-D",
        "--disease_set",
        type=str,
        default="gmb",
        choices=dt.valid_datasets,
    )

    args = parser.parse_args()
    return {
        "network": args.network,
        "disease_set": args.disease_set,
    }


def main() -> None:
    params = parse_args()
    network = params["network"]
    disease_set = params["disease_set"]
    _, _, seeds_by_disease = dt.load_dataset(
        disease_set,
        network,
        dt.FilterGCC.TRUE,
    )
    print("\n".join(f"{k} (num_seeds:{len(v)})" for k, v in seeds_by_disease.items()))
    print(f"{len(list(seeds_by_disease.keys()))} diseases available.")


if __name__ == "__main__":
    main()
