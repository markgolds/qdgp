"""Handle graph and disease loading."""

import csv
import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, get_args

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

VALID_DATASETS = Literal["gmb", "ot", "dgn"]
_valid_datasets: List[str] = list(get_args(VALID_DATASETS))

VALID_NETWORKS = Literal[
    "gmb",  # https://pubmed.ncbi.nlm.nih.gov/25853560/
    "biogrid",  # https://academic.oup.com/bib/article-abstract/22/5/bbab066/6189770
    "string",
    "iid",
    "apid",
    "hprd",
    "wl",  # https://pubmed.ncbi.nlm.nih.gov/34132494/
]
_valid_networks: List[str] = list(get_args(VALID_NETWORKS))


class GraphFilterMethod(Enum):
    """Controls whether or not to use only the gcc of the network."""

    WHOLE = 0  # use the whole graph
    GCC = 1  # keep only greatest connected component


def _build_graph(
    g_df: pd.DataFrame,
    filter_method: GraphFilterMethod,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Turn pandas dataframe into graph.

    Args:
    ----
    g_df: Dataframe with "source" and "target" columns
    filter_method: How to filter the ppi network

    Return: Networkx graph and code_dict which maps gene ids to graph nodes

    """
    if filter_method == GraphFilterMethod.GCC:
        G_ = nx.from_pandas_edgelist(g_df)
        Gcc = sorted(nx.connected_components(G_), key=len, reverse=True)
        G0 = G_.subgraph(Gcc[0])
        gcc_nodes = list(G0.nodes())
        g_df = pd.DataFrame(g_df[g_df.source.isin(gcc_nodes)])
        g_df = pd.DataFrame(g_df[g_df.target.isin(gcc_nodes)])
    # Map gene ids to {0, 1, ..., num_nodes}:
    ids = pd.concat([g_df["source"], g_df["target"]])
    vals = ids.astype("category").cat.codes.to_numpy()
    code_dict = {}
    for gid, v in zip(ids, vals):
        code_dict[gid] = v
    g_df = g_df.replace({"source": code_dict})
    g_df = g_df.replace({"target": code_dict})
    G = nx.from_pandas_edgelist(g_df)
    return G, code_dict


def build_graph_wl(
    data_dir: str,
    filter_method: GraphFilterMethod,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Read csv file and convert it to a networkx graph.

    Args:
    ----
    data_dir: path to data file
    filter_method: How to filter the ppi network

    Return: Networkx graph and code_dict which maps gene ids to graph nodes

    """
    wl_df = pd.read_csv(
        f"{data_dir}/PPI202207.txt",
        delim_whitespace=True,
        skiprows=[237433 - 1],  # header on this row
        header=None,
        usecols=[0, 2],
        dtype=int,
    )
    wl_df = wl_df[[0, 2]]
    wl_df.columns = ["source", "target"]
    G, code_dict = _build_graph(wl_df, filter_method)
    return G, code_dict


def build_graph_gmb(
    data_dir: str,
    filter_method: GraphFilterMethod,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Read csv file and convert it to a networkx graph.

    Args:
    ----
    data_dir: path to data file
    filter_method: Use whole graph (False) or only giant component (True)

    Return: Networkx graph and code_dict which maps gene ids to graph nodes

    """
    gmb_df = pd.read_csv(
        f"{data_dir}/gmb/pcbi.1004120.s003.tsv",
        delimiter="\t",
        dtype={"gene_ID_1": int, "gene_ID_2": int},
    )
    gmb_df.columns = [s.strip() for s in gmb_df.columns]
    gmb_df = gmb_df[["gene_ID_1", "gene_ID_2"]]
    gmb_df.columns = ["source", "target"]
    G, code_dict = _build_graph(gmb_df, filter_method)
    return G, code_dict


def build_graph_loami(
    data_dir: str,
    network: str,
    filter_method: GraphFilterMethod,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Read csv file and convert it to a networkx graph.

    Args:
    ----
    data_dir: path to data file
    network: name of ppi network
    filter_method: How to filter the ppi network

    Return: Networkx graph and code_dict which maps gene ids to graph nodes

    """
    if network not in _valid_networks:
        e = f"{network} not recognized. Must be one of {_valid_networks}"
        raise ValueError(e)
    if network == "string":
        G = nx.read_graphml(f"{data_dir}/loami/STRING.graphml")
    elif network == "iid":
        G = nx.read_graphml(f"{data_dir}/loami/IID.graphml")
    elif network == "apid":
        G = nx.read_graphml(f"{data_dir}/loami/APID.graphml")
    elif network == "hprd":
        G = nx.read_graphml(f"{data_dir}/loami/HPRD.graphml")
    else:
        G = nx.read_graphml(f"{data_dir}/loami/BIOGRID.graphml")
    if filter_method == GraphFilterMethod.GCC:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
    GG = nx.convert_node_labels_to_integers(G)
    code_dict = {}
    for node in GG.nodes():
        code_dict[int(GG.nodes[node]["GeneID"])] = int(node)
    return GG, code_dict


def process_diseases_gmb(
    node_list: List[int],
    code_dict: Dict[int, int],
    data_dir: str,
) -> pd.DataFrame:
    """Read disease file and return a dataframe associating diseases and gene nodes.

    Args:
    ----
    node_list: list of nodes from the graph
    code_dict: dictionary mapping gene ids to node labels
    data_dir: string reprsenting data path

    Return: pandas dataframe with "disease" and "gene" columns

    """
    df_rows = []
    # with open(f"{data_dir}/menche/pcbi.1004120.s004.tsv", newline="") as csvfile:
    with Path(f"{data_dir}/gmb/pcbi.1004120.s004.tsv").open(newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        next(reader, None)  # skip the headers
        for row in reader:
            disease = " ".join(row[:-1])
            disease = disease.rstrip()
            genes = row[-1]
            for gid in genes.split("/"):
                try:
                    if code_dict[int(gid)] in node_list:
                        df_rows.append([disease, code_dict[int(gid)]])
                    else:
                        logger.info("disease/node problem.")
                except Exception:
                    logger.info("%s skipped. %s not present in graph.", disease, gid)
    df_data = pd.DataFrame(df_rows)
    df_data.columns = ["disease", "gene"]
    return df_data[df_data.groupby("disease")["disease"].transform("size").ge(15)]


def code_mapper(
    row: pd.Series,
    code_dict: Dict[int, int],
    col_name: str,
) -> Optional[int]:
    """Apply code_dict to row of DataFrame to change entry in col_name.

    Args:
    ----
    row: Row object from the DataFrame
    code_dict: Dictionary that maps original gene names to node ids
    col_name: Name of the column holding the data

    """
    if int(row[col_name]) in code_dict:
        return code_dict[int(row[col_name])]
    return None


def process_diseases_ot(
    node_list: List[int],
    code_dict: Dict[int, int],
    data_dir: str,
) -> pd.DataFrame:
    """Read disease file and return a dataframe associating diseases and gene nodes.

    Args:
    ----
    node_list: list of nodes from the graph, ignored in this function
    code_dict: dictionary mapping gene ids to node labels
    data_dir: string reprsenting data path

    Returns a pandas dataframe with "disease" and "gene" columns

    """
    ot_df = pd.read_csv(
        f"{data_dir}/OpenTargetsIDs.csv",
        delimiter=",",
    )
    ot_df = ot_df.drop_duplicates()
    threshold = 0.6
    ot_df = ot_df[ot_df.score >= threshold]
    ot_df = ot_df[["id", "Disease_Name"]]
    ot_df = ot_df.dropna()
    ot_df.columns = ["gene", "disease"]
    ot_df["gene"] = ot_df.apply(lambda row: code_mapper(row, code_dict, "gene"), axis=1)
    ot_df = ot_df[["disease", "gene"]].dropna()
    ot_df["gene"] = ot_df["gene"].astype(int)
    # Keep only diseases ith at least 15 seeds
    return ot_df[ot_df.groupby("disease")["disease"].transform("size").ge(15)]


def process_diseases_dgn(
    node_list: List[int],
    code_dict: Dict[int, int],
    data_dir: str,
) -> pd.DataFrame:
    """Read disease file and return a dataframe associating diseases and gene nodes.

    Args:
    ----
    node_list: list of nodes from the graph, ignored in this function
    code_dict: dictionary mapping gene ids to node labels
    data_dir: string reprsenting data path

    Returns a pandas dataframe with "disease" and "gene" columns

    """
    dgn_df = pd.read_csv(
        f"{data_dir}/DisGenet_all.csv",
        delimiter=",",
        usecols=["score", "geneId", "diseaseName", "type", "EL", "EI", "source", "DSI"],
    )

    dgn_df = dgn_df.drop_duplicates()
    dgn_df = dgn_df[(dgn_df.type == "disease")]
    score_threshold = 0.3
    dsi_threshold = 0.5
    dgn_df = dgn_df[dgn_df.score >= score_threshold]
    dgn_df = dgn_df[dgn_df.DSI >= dsi_threshold]
    dgn_df = dgn_df[["geneId", "diseaseName"]]
    dgn_df = dgn_df.dropna()
    dgn_df.columns = ["gene", "disease"]
    dgn_df["gene"] = dgn_df.apply(
        lambda row: code_mapper(row, code_dict, "gene"),
        axis=1,
    )
    dgn_df = dgn_df[["disease", "gene"]].dropna()
    dgn_df["gene"] = dgn_df["gene"].astype(int)
    dgn_df = dgn_df.drop_duplicates()
    # Keep only diseases ith at least 15 seeds
    return dgn_df[dgn_df.groupby("disease")["disease"].transform("size").ge(15)]


def get_disease_nodes(
    node_list: List,
    code_dict: Dict[int, int],
    data_dir: str,
    method: Callable,
) -> Dict[str, List]:
    """Build dictionary mapping disease names to node labels.

    Args:
    ----
    node_list: List of nodes in the graph
    code_dict: Maps gene ids to node ids
    data_dir: Path where data is stored
    method: Method for reading the desired data set.

    """
    df_data = method(node_list, code_dict, data_dir)
    return df_data.groupby("disease")["gene"].agg(list).to_dict()


def load_dataset(
    disease_set: str,  # VALID_DATASETS = "gmb",
    network: str,  # VALID_NETWORKS = "biogrid",
    filter_method: GraphFilterMethod,
) -> Tuple[nx.Graph, Dict[int, int], Dict[str, List[int]]]:
    """Build network and disease data from raw files.

    Args:
    ----
    disease_set: Name of the data set
    network: Name of the ppi network
    filter_method: How to filter the ppi network

    """
    if disease_set not in _valid_datasets:
        e = f"{disease_set} not recognized. Must be one of {_valid_datasets}"
        raise ValueError(e)
    if network not in _valid_networks:
        e = f"{network} not recognized. Must be one of {_valid_networks}"
        raise ValueError(e)
    data_dir = "data"

    # Load the network:
    if network in [
        "biogrid",
        "string",
        "iid",
        "apid",
        "hprd",
    ]:
        G, code_dict = build_graph_loami(
            data_dir,
            network=network,
            filter_method=filter_method,
        )
    elif network == "wl-ppi":
        G, code_dict = build_graph_wl(data_dir, filter_method=filter_method)
    else:
        G, code_dict = build_graph_gmb(data_dir, filter_method=filter_method)

    # Load the disease set:
    if disease_set == "ot":
        func = process_diseases_ot
    elif disease_set == "dgn":
        func = process_diseases_dgn
    else:  # disease_set == "gmb":
        func = process_diseases_gmb

    disease_nodes_by_disease = get_disease_nodes(
        list(G.nodes()),
        code_dict,
        data_dir,
        func,
    )

    return G, code_dict, disease_nodes_by_disease


def get_graph(
    network: VALID_NETWORKS = "biogrid",
    filter_method: GraphFilterMethod = GraphFilterMethod.GCC,
) -> nx.Graph:
    """Build network and disease data from raw files.

    Args:
    ----
    network: Name of the PPI network
    filter_method: How to filter the ppi network

    """
    if network not in _valid_networks:
        e = f"{network} not recognized. Must be one of {_valid_networks}"
        raise ValueError(e)
    data_dir = "../data"

    if network in [
        "biogrid",
        "string",
        "iid",
        "apid",
        "hprd",
    ]:
        G, code_dict = build_graph_loami(data_dir, network=network, gcc=gcc)
    elif network == "wl":
        G, code_dict = build_graph_wl(data_dir, gcc=gcc)
    else:
        G, code_dict = build_graph_gmb(data_dir, gcc=gcc)
    return G
