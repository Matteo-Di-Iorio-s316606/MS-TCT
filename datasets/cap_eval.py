import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CAPEvalDataset(Dataset):
    def __init__(
        self,
        features_dir,
        label_dir,
        val_list,
        window_size=16,
    ):
        self.features_dir = features_dir
        self.label_dir = label_dir
        self.window_size = window_size

        with open(val_list, "r") as f:
            self.vids = [x.strip() for x in f.readlines() if x.strip()]

        # build map vid -> txt path
        self.vid2txt = {}
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            base = os.path.splitext(fname)[0]
            vid = base.rsplit("_", 1)[1]
            self.vid2txt[vid] = os.path.join(label_dir, fname)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        feat_path = os.path.join(self.features_dir, f"{vid}.npy")
        x = np.load(feat_path)  # (T, D)
        T = x.shape[0]

        txt_path = self.vid2txt[vid]
        segments = []

        with open(txt_path, "r") as f:
            for line in f:
                a = line.strip().split()
                if len(a) < 3:
                    continue
                cid = int(a[0])
                sf = int(a[1])
                ef = int(a[2])

                # convert frame → feature index
                s = sf // self.window_size
                e = ef // self.window_size
                s = min(max(s, 0), T - 1)
                e = min(max(e, 0), T - 1)

                segments.append([cid, s, e])

        return {
            "vid": vid,
            "feat": torch.from_numpy(x).float(),
            "segments": torch.tensor(segments)
        }