import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from typing import List, Tuple, Dict
class SpeakerEmbeddingDataset(Dataset):
    def __init__(
        self,
        root: str,
        known: List[int] = None,
        mask: str = 'all'
    ) -> None:
        self.data = []
        self.targets = []
        self.known = set(known) if known is not None else None
        self.mask = mask

        # Load the embeddings
        self._load_data(root)

    def _load_data(self, root):
        for speaker_id in os.listdir(root):
            speaker_id_int = int(speaker_id)
            speaker_folder = os.path.join(root, speaker_id)
            files = sorted(os.listdir(speaker_folder))

            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(speaker_folder, file)
                    embedding = np.load(file_path).squeeze()
                    assert embedding.shape == (192,), f"Expected embedding shape (192,), but got {embedding.shape}"

                    if self.known is None or \
                       (self.mask == 'known' and speaker_id_int in self.known) or \
                       (self.mask == 'unknown' and speaker_id_int not in self.known):
                        self.data.append(embedding)
                        self.targets.append(speaker_id_int)
        if len(self.data) > 0:
            self.data = np.vstack(self.data)

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        embedding, target = self.data[index], self.targets[index]
        return embedding, target

    def __len__(self) -> int:
        return len(self.data)


from typing import List, Dict
from torch.utils.data import DataLoader

class SpeakerDataloader:
    def __init__(
        self, 
        known: List[int],
        train_root: str,
        test_root: str,
        use_gpu: bool = True, 
        num_workers: int = 8, 
        batch_size: int = 128
    ):
        self.known = known
        self.num_classes = len(known)
        print('num classes:',len(known))
        # Loaders for the training set
        trainset = SpeakerEmbeddingDataset(train_root, known=self.known, mask='known')
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)

        # Loaders for the test set - target speakers
        testset_known = SpeakerEmbeddingDataset(test_root, known=self.known, mask='known')
        self.test_loader = DataLoader(testset_known, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)

        # Loaders for the test set - outliers
        testset_unknown = SpeakerEmbeddingDataset(test_root, known=self.known, mask='unknown')
        self.out_loader = DataLoader(testset_unknown, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)


        print('Train: ', len(trainset), 'Test Known: ', len(testset_known), 'Test Unknown: ', len(testset_unknown))