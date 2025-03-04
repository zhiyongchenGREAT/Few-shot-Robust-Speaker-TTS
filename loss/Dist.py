import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dist(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_centers=1,
        feat_dim=2,
        embedding_dir=None,  # Path to root directory containing speaker subfolders
        init='random'
    ):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            # Random initialization
            self.centers = nn.Parameter(
                0.1 * torch.randn(num_classes * num_centers, self.feat_dim)
            )

        elif init == 'zero':
            # All zeros
            self.centers = nn.Parameter(
                torch.zeros(num_classes * num_centers, self.feat_dim)
            )

        elif init == 'mean_all':
            print('#!')
            # 1) Read all .npy files across all speakers
            # 2) Compute a single global mean
            # 3) Initialize every row of self.centers with that global mean
            if embedding_dir is None:
                raise ValueError("'embedding_dir' must be provided for 'mean_all' init")

            # Gather all embeddings into a Python list
            all_embs = []
            for speaker_id_str in os.listdir(embedding_dir):
                speaker_path = os.path.join(embedding_dir, speaker_id_str)
                if not os.path.isdir(speaker_path):
                    continue
                for npy_file in os.listdir(speaker_path):
                    if npy_file.endswith('.npy'):
                        full_path = os.path.join(speaker_path, npy_file)
                        emb = np.load(full_path)  # shape = (feat_dim,) or something similar
                        all_embs.append(emb)

            if len(all_embs) == 0:
                raise ValueError(f"No .npy files found under '{embedding_dir}'")

            # Convert list to a single numpy array (N x feat_dim)
            all_embs = np.stack(all_embs, axis=0)
            global_mean = np.mean(all_embs, axis=0)  # shape = (feat_dim,)

            # Convert numpy -> torch
            global_mean_t = torch.from_numpy(global_mean).float()

            # Repeat for each center (num_classes * num_centers)
            repeated_mean = global_mean_t.unsqueeze(0).repeat(self.num_classes*self.num_centers, 1)
            self.centers = nn.Parameter(repeated_mean)

        elif init == 'rotate':
            # 1) For each speaker, compute the mean embedding.
            # 2) Initialize that speaker's center(s) with the negative of its mean embedding.
            if embedding_dir is None:
                raise ValueError("'embedding_dir' must be provided for 'rotate' init")

            speaker_means = []
            # We assume speaker subfolders are named 0, 1, 2, ... up to num_classes-1
            # If your subfolder names differ, adjust accordingly.
            for speaker_id in range(self.num_classes):
                speaker_path = os.path.join(embedding_dir, str(speaker_id))
                if not os.path.isdir(speaker_path):
                    raise ValueError(f"Directory for speaker '{speaker_id}' not found.")

                embs = []
                for npy_file in os.listdir(speaker_path):
                    if npy_file.endswith('.npy'):
                        full_path = os.path.join(speaker_path, npy_file)
                        emb = np.load(full_path)
                        embs.append(emb)

                if len(embs) == 0:
                    raise ValueError(f"No .npy files found in {speaker_path}")

                # Compute mean embedding for this speaker
                embs = np.stack(embs, axis=0)  # shape = (N, feat_dim)
                spk_mean = np.mean(embs, axis=0)  # (feat_dim,)
                speaker_means.append(spk_mean)

            # Convert to torch, shape = (num_classes, feat_dim)
            speaker_means_t = torch.from_numpy(np.stack(speaker_means, axis=0)).float()
            # Rotate by 180 => multiply by -1
            speaker_means_rotated = -1.0 * speaker_means_t  # shape = (num_classes, feat_dim)

            # If num_centers > 1, you can replicate or do something else for each center.
            # Here, we'll just replicate:
            # shape => (num_classes, num_centers, feat_dim)
            speaker_means_rotated = speaker_means_rotated.unsqueeze(1).repeat(1, self.num_centers, 1)

            # Flatten to (num_classes * num_centers, feat_dim)
            speaker_means_rotated = speaker_means_rotated.view(-1, self.feat_dim)
            self.centers = nn.Parameter(speaker_means_rotated)

        elif init == 'rotate_norm':
            # 1) For each speaker, compute the mean embedding.
            # 2) Initialize that speaker's center(s) with the negative of its mean embedding.
            if embedding_dir is None:
                raise ValueError("'embedding_dir' must be provided for 'rotate' init")

            speaker_means = []
            # We assume speaker subfolders are named 0, 1, 2, ... up to num_classes-1
            # If your subfolder names differ, adjust accordingly.
            for speaker_id in range(self.num_classes):
                speaker_path = os.path.join(embedding_dir, str(speaker_id))
                if not os.path.isdir(speaker_path):
                    raise ValueError(f"Directory for speaker '{speaker_id}' not found.")

                embs = []
                for npy_file in os.listdir(speaker_path):
                    if npy_file.endswith('.npy'):
                        full_path = os.path.join(speaker_path, npy_file)
                        emb = np.load(full_path)
                        embs.append(emb)

                if len(embs) == 0:
                    raise ValueError(f"No .npy files found in {speaker_path}")

                # Compute mean embedding for this speaker
                embs = np.stack(embs, axis=0)  # shape = (N, feat_dim)
                spk_mean = np.mean(embs, axis=0)  # (feat_dim,)
                speaker_means.append(spk_mean)

            # Convert to torch, shape = (num_classes, feat_dim)
            speaker_means_t = torch.from_numpy(np.stack(speaker_means, axis=0)).float()

            # Rotate by 180 => multiply by -1
            speaker_means_rotated = -1.0 * speaker_means_t  # shape = (num_classes, feat_dim)

            # Optionally, normalize the embeddings so each is unit-length
            # shape = (num_classes, feat_dim)
            norms = torch.norm(speaker_means_rotated, p=2, dim=1, keepdim=True)
            # Safeguard against zero norm
            norms = torch.clamp(norms, min=1e-8)
            speaker_means_rotated = 0.1 * speaker_means_rotated / norms

            # If num_centers > 1, replicate or otherwise handle multiple centers per class
            # shape => (num_classes, num_centers, feat_dim)
            speaker_means_rotated = speaker_means_rotated.unsqueeze(1).repeat(1, self.num_centers, 1)

            # Flatten to (num_classes * num_centers, feat_dim)
            speaker_means_rotated = speaker_means_rotated.view(-1, self.feat_dim)
            
            self.centers = nn.Parameter(speaker_means_rotated)

        elif init == 'centered':
            # 1) For each speaker, compute the mean embedding.
            # 2) Initialize that speaker's center(s) with the speaker's own mean embedding.
            if embedding_dir is None:
                raise ValueError("'embedding_dir' must be provided for 'centered' init")

            speaker_means = []
            # We assume speaker subfolders are named 0, 1, 2, ... up to num_classes-1
            # If your subfolder names differ, adjust accordingly.
            for speaker_id in range(self.num_classes):
                speaker_path = os.path.join(embedding_dir, str(speaker_id))
                if not os.path.isdir(speaker_path):
                    raise ValueError(f"Directory for speaker '{speaker_id}' not found.")

                embs = []
                for npy_file in os.listdir(speaker_path):
                    if npy_file.endswith('.npy'):
                        full_path = os.path.join(speaker_path, npy_file)
                        emb = np.load(full_path)
                        embs.append(emb)

                if len(embs) == 0:
                    raise ValueError(f"No .npy files found in '{speaker_path}'")

                # Compute mean embedding for this speaker
                embs = np.stack(embs, axis=0)  # shape = (N, feat_dim)
                spk_mean = np.mean(embs, axis=0)  # (feat_dim,)
                speaker_means.append(spk_mean)

            # Convert to torch, shape = (num_classes, feat_dim)
            speaker_means_t = torch.from_numpy(np.stack(speaker_means, axis=0)).float()

            # If num_centers > 1, replicate (or handle multiple centers differently if desired).
            # shape => (num_classes, num_centers, feat_dim)
            speaker_means_centered = speaker_means_t.unsqueeze(1).repeat(1, self.num_centers, 1)

            # Flatten to (num_classes * num_centers, feat_dim)
            speaker_means_centered = speaker_means_centered.view(-1, self.feat_dim)

            # Register as parameter
            self.centers = nn.Parameter(speaker_means_centered)

        elif init == 'centered_norm':
            # 1) For each speaker, compute the mean embedding.
            # 2) Initialize that speaker's center(s) with the speaker's own mean embedding.
            if embedding_dir is None:
                raise ValueError("'embedding_dir' must be provided for 'centered' init")

            speaker_means = []
            # We assume speaker subfolders are named 0, 1, 2, ... up to num_classes-1
            # If your subfolder names differ, adjust accordingly.
            for speaker_id in range(self.num_classes):
                speaker_path = os.path.join(embedding_dir, str(speaker_id))
                if not os.path.isdir(speaker_path):
                    raise ValueError(f"Directory for speaker '{speaker_id}' not found.")

                embs = []
                for npy_file in os.listdir(speaker_path):
                    if npy_file.endswith('.npy'):
                        full_path = os.path.join(speaker_path, npy_file)
                        emb = np.load(full_path)
                        embs.append(emb)

                if len(embs) == 0:
                    raise ValueError(f"No .npy files found in '{speaker_path}'")

                # Compute mean embedding for this speaker
                embs = np.stack(embs, axis=0)  # shape = (N, feat_dim)
                spk_mean = np.mean(embs, axis=0)  # (feat_dim,)
                speaker_means.append(spk_mean)

            # Convert to torch, shape = (num_classes, feat_dim)
            speaker_means_t = torch.from_numpy(np.stack(speaker_means, axis=0)).float()

            # Optionally, normalize the embeddings so each is unit-length
            # shape = (num_classes, feat_dim)
            norms = torch.norm(speaker_means_t, p=2, dim=1, keepdim=True)
            # Safeguard against zero norm
            norms = torch.clamp(norms, min=1e-8)
            speaker_means_t = 0.1 * speaker_means_t / norms

            # If num_centers > 1, replicate (or handle multiple centers differently if desired).
            # shape => (num_classes, num_centers, feat_dim)
            speaker_means_centered = speaker_means_t.unsqueeze(1).repeat(1, self.num_centers, 1)

            # Flatten to (num_classes * num_centers, feat_dim)
            speaker_means_centered = speaker_means_centered.view(-1, self.feat_dim)

            # Register as parameter
            self.centers = nn.Parameter(speaker_means_centered)

        elif init == 'sphere':
            # Let all centers be distributed on a sphere of radius 0.1
            # across a (num_classes * num_centers)-element set.

            # Total number of centers
            total_centers = self.num_classes * self.num_centers

            # 1) Sample random Gaussian vectors
            centers_t = torch.randn(total_centers, self.feat_dim)

            # 2) Normalize each vector to have norm=1 (unit sphere)
            norms = torch.norm(centers_t, p=2, dim=1, keepdim=True)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            centers_t = centers_t / norms

            # 3) Scale by the desired sphere radius (0.1)
            centers_t = centers_t * 0.1

            # 4) Register as learnable parameters
            self.centers = nn.Parameter(centers_t)

        else:
            raise ValueError(f"Unknown init type '{init}'")


    def forward(self, features, center=None, metric='l2'):
        # print(f"[DEBUG] Features shape: {features.shape}")
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                # print(f"[DEBUG] Center shape: {center.shape}")
                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers 
            else:
                center = center 
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2) 

        return dist
