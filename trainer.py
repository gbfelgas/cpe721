"""
Authors:
    Maria Gabriella Andrade Felgas
    Pedro Gil Oliveira de Magalhães Couto

Main python script. It initializes a Lightning Module of the original model
and fits it with a G1 News dataset. The parameters can be changed in the constants of this file.
"""

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.module import LightningModuleMLP
from src.data import g1_news_preprocess, EmbeddingsDataset

EMBEDDINGS_PATH = "data/datasets/g1-news/articles_embeddings.pickle"
METADATA_PATH = "data/datasets/g1-news/articles_metadata.csv"
ENTRY_COLUMN = "article_id"
CLASS_COLUMN = "category_id"
CLASS_THRESHOLD = 0.01

TRAIN_SIZE = 0.8

BATCH_SIZE = 512
HIDDEN_DIM = 64
DROPOUT = 0.1
NUM_LAYERS = 2
LEARNING_RATE = 0.0008
NUM_WORKERS = 8
SHUFFLE = True
RANDOM_STATE = 42

seed_everything(RANDOM_STATE)

def main(args):

    X, y, emb_dim, n_classes = g1_news_preprocess(
        EMBEDDINGS_PATH,
        METADATA_PATH,
        ENTRY_COLUMN,
        CLASS_COLUMN,
        CLASS_THRESHOLD
    )

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, random_state=RANDOM_STATE, train_size=0.5)

    train_dataset = EmbeddingsDataset(X_train, y_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=SHUFFLE
    )

    val_dataset = EmbeddingsDataset(X_val, y_val)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=SHUFFLE
    )

    test_dataset = EmbeddingsDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=SHUFFLE
    )

    model = LightningModuleMLP(
        input_dim=emb_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=n_classes,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        num_layers=NUM_LAYERS
    )

    trainer = Trainer.from_argparse_args(
        args,
        gpus=1,
        val_check_interval=1.0,
        log_every_n_steps=1,
        # track_grad_norm=2
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)