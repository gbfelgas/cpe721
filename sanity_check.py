from src.data import EmbeddingsDataset
from torch.utils.data import DataLoader

BATCH_SIZE = 16
NUM_WORKERS = 0
SHUFFLE = False

dataset = EmbeddingsDataset()
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=SHUFFLE
)

for (x, y) in dataloader:
    print(x.shape)
    print(y.shape)
    break