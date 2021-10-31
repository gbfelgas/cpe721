"""
Authors:
    Maria Gabriella Andrade Felgas
    Pedro Gil Oliveira de Magalhães Couto

Main python script. It initializes a Lightning Module of the original model
and fits it with G1 News dataset. The parameters can be changed in the constants of this file.
"""

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
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
NUM_LAYERS = 2

SHUFFLE = True
NUM_WORKERS = 0
RANDOM_STATE = 42

grid = {
    "batch_size": [256, 512, 1024],
    "hidden_dim": [32, 64, 128],
    "dropout": [0., 0.05, 0.08, 0.1],
    "learning_rate":  [0.0005, 0.001, 0.005]
}

seed_everything(RANDOM_STATE)

def main(args):
    
    X, y, emb_dim, n_classes = g1_news_preprocess(
        EMBEDDINGS_PATH,
        METADATA_PATH,
        ENTRY_COLUMN,
        CLASS_COLUMN,
        CLASS_THRESHOLD
    )

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        random_state=RANDOM_STATE,
        train_size=0.5
    )

    print(f"Number of classes considered: {n_classes}")
    print(f"Dataset length: {len(X)}")

    for b in grid["batch_size"]:
        batch_size = b

        train_dataset = EmbeddingsDataset(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE
        )

        val_dataset = EmbeddingsDataset(X_val, y_val)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE
        )

        test_dataset = EmbeddingsDataset(X_test, y_test)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE
        )

        for h in grid["hidden_dim"]:
            hidden_dim = h
            for d in grid["dropout"]:
                dropout = d
                for lr in grid["learning_rate"]:
                    learning_rate = lr

                    model = LightningModuleMLP(
                        input_dim=emb_dim,
                        hidden_dim=hidden_dim,
                        output_dim=n_classes,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        num_layers=NUM_LAYERS
                    )

                    tb_logger = TensorBoardLogger(
                        "data/logs/grid_logs_with_acc",
                        name=f"model_{batch_size}_{hidden_dim}_{dropout}_{learning_rate}"
                    )

                    trainer = Trainer.from_argparse_args(
                        args,
                        gpus=1,
                        val_check_interval=1.0,
                        log_every_n_steps=1,
                        default_root_dir="data/logs/",
                        logger=tb_logger,
                        # track_grad_norm=2
                    )
                    trainer.fit(model, train_dataloader, val_dataloader)
                    trainer.test(model, test_dataloader)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
