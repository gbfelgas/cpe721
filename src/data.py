"""
This file contains all data preprocessing and selection.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def g1_news_preprocess(
    embeddings_path: str,
    metadata_path: str,
    entry_column: str,
    class_column: str,
    class_threshold: float
):
    """
    Cast data types to tensors and selects classes that
    have more than <class_threshold> percent of dataset samples.

    Args:
        embeddings_path (str):
            Input embeddings dataset path, it contains the embeddings of all entries.
            (size: (number of entries, embeddings dimension))
        metadata_path (str):
            Metadata dataset path, it contains metadata of all entries
            and assumes classes are storage in <class_column>.
            (size: (number of entries, ...))
        entry_column (str):
            Column of metadata dataset where entry id is stored.
        class_column (str):
            Column of metadata dataset where class id is stored.
        class_threshold (float):
            Thrshold to select classes and samples. Only classes with more than
            total number of samples * class_threshold will be selected.

    Returns:
        X (np.ndarray):
            Filtered input dataset, with only selected classes entries.
            (size: (number of entries from the selected classes, embeddings dimension))
        y (np.ndarray):
            Filtered output dataset, with only selected classes entries.
            (size: (number of entries from the selected classes,))
    """
    
    embeddings = pd.read_pickle(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    emb_size = len(embeddings)
    met_size = len(metadata)

    if emb_size != met_size:
        raise ValueError(
            f"Embeddings and metadata must have the same number of entries!\n\
                Number of embeddings entries = {emb_size}\n\
                Number of metadata entries = {met_size}"
        )

    grouped_metadata = metadata.groupby(by=[class_column]).count()
    selected_classes = grouped_metadata[grouped_metadata[entry_column] >= (class_threshold * met_size)].index
    n_classes = len(selected_classes)
    map_classes = {selected_classes[i]: i for i in range(n_classes)}

    filtered_metadata = metadata[metadata[class_column].isin(selected_classes)].copy()
    filtered_metadata[class_column] = filtered_metadata[class_column].apply(lambda x: map_classes[x])
    X = embeddings[filtered_metadata[entry_column].values, :]
    y = filtered_metadata[class_column].values
    emb_dim = X.shape[-1]

    return X, y, emb_dim, n_classes
    

class EmbeddingsDataset(Dataset):

    def __init__(self, X: torch.tensor, y: torch.tensor):
        super().__init__()
        self.X, self.y = torch.tensor(X), torch.tensor(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)