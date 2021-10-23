"""
This file contains all data preprocessing and selection.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EmbeddingsDataset(Dataset):

    def __init__(
        self,
        embeddings_path: str = 'data/datasets/g1-news/articles_embeddings.pickle',
        metadata_path: str = 'data/datasets/g1-news/articles_metadata.csv',
        entry_column: str = "article_id",
        class_column: str = "category_id",
        class_threshold: float = 0.01
    ):
        super().__init__()

        embeddings = pd.read_pickle(embeddings_path)
        metadata = pd.read_csv(metadata_path)
        self.X, self.y = self.preprocess(
            embeddings,
            metadata,
            entry_column,
            class_column,
            class_threshold
        )

    @staticmethod
    def preprocess(
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        entry_column: str,
        class_column: str,
        class_threshold: float
    ):
        """
        Cast data types to tensors and selects classes that
        have more than <class_threshold> percent of dataset samples.

        Args:
            embeddings (np.ndarray):
                Input embeddings dataset, it contains the embeddings of all entries.
                (size: (number of entries, embeddings dimension))
            metadata (pd.DataFrame):
                Metadata dataset, it contains metadata of all entries
                and assumes classes are storage in <class_column>.
                (size: (number of entries, ...))

        Returns:
            X (torch.tensor):
                Filtered input dataset, with only selected classes entries.
                (size: (number of entries from the selected classes, embeddings dimension))
            y (torch.tensor):
                Filtered output dataset, with only selected classes entries.
                (size: (number of entries from the selected classes,))
        """

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
        print(f"Total number of classes = {metadata[class_column].nunique()}")
        print(f"Number of selected classes = {len(selected_classes)}")

        filtered_metadata = metadata[metadata[class_column].isin(selected_classes)]

        X = torch.tensor(embeddings[filtered_metadata[entry_column].values, :])
        y = torch.tensor(filtered_metadata[class_column].values)

        return X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)