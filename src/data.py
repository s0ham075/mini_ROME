import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

REMOTE_URL = "https://rome.baulab.info/data/dsets/known_1000.json"

class KnownsDataset(Dataset):
    """
    Dataset for loading 'known_1000.json'.
    Downloads the file if it's not already present in the data directory.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.known_loc = self.data_dir / "known_1000.json"

        # Download if file doesn't exist
        if not self.known_loc.exists():
            print(f"{self.known_loc} not found. Downloading from {REMOTE_URL}...")
            self.data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, self.known_loc)

        # Load data
        with open(self.known_loc, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
