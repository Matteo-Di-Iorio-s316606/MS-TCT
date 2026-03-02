# cap_dataloader.py
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

def build_vid2txt(label_new_dir):
    vid2txt = {}
    for p in glob.glob(os.path.join(label_new_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(p))[0]
        if "_" not in base:
            continue
        vid = base.rsplit("_", 1)[1]          # <category>_<vid>
        vid2txt[vid] = p
    return vid2txt

class CAP(TorchDataset):
    """
    Signature compatibile con load_data():
      Dataset(split, 'training'/'testing', root, batch_size, classes, num_clips, skip)

    Dove root = args.rgb_root (dir delle feature .npy)
    e label dir lo prendiamo da ENV per non toccare train.py:
      export CAP_LABEL_NEW_DIR="/.../CAP_label_NEW"
      export CAP_WINDOW_SIZE="16"
    """
    def __init__(self, split_file, mode, root, batch_size, classes, num_clips, skip):
        self.mode = mode
        self.root = root
        self.batch_size = batch_size
        self.classes = int(classes)
        self.num_clips = int(num_clips)
        self.skip = int(skip)

        self.label_dir = os.environ.get("CAP_LABEL_NEW_DIR", None)
        if self.label_dir is None:
            raise RuntimeError("Set env var CAP_LABEL_NEW_DIR to CAP_label_NEW path.")

        self.window_size = int(os.environ.get("CAP_WINDOW_SIZE", "16"))

        with open(split_file, "r") as f:
            self.vids = [ln.strip() for ln in f if ln.strip()]

        self.vid2txt = build_vid2txt(self.label_dir)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        feat_path = os.path.join(self.root, f"{vid}.npy")
        x = np.load(feat_path)  # (T, D)
        T, D = x.shape

        # unisize pad/crop
        L = self.num_clips
        mask = np.zeros((L,), dtype=np.float32)

        if T >= L:
            x_u = x[:L]
            mask[:] = 1.0
            T_eff = L
        else:
            x_u = np.zeros((L, D), dtype=x.dtype)
            x_u[:T] = x
            mask[:T] = 1.0
            T_eff = T

        # labels: (L, classes)
        labels = np.zeros((L, self.classes), dtype=np.float32)

        # read segments from CAP_label_NEW: "<category>_<vid>.txt"
        txt_path = self.vid2txt.get(vid, None)
        if txt_path is not None:
            with open(txt_path, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    a = ln.split()
                    if len(a) < 3:
                        continue
                    cid = int(a[0])
                    sf = int(a[1])
                    ef = int(a[2])

                    s = sf // self.window_size
                    e = ef // self.window_size
                    s = max(0, min(s, L-1))
                    e = max(0, min(e, L-1))
                    if e < s:
                        s, e = e, s
                    labels[s:e+1, cid] = 1.0

        # hm: metti zero e usa -beta_l 0 in eval-only (o anche training)
        # shape: deve matchare out_hm. Non sappiamo shape, quindi metti un placeholder e poi beta_l=0.
        hm = np.zeros((1,), dtype=np.float32)

        # inputs devono essere (T, C, 1, 1) come Charades
        inputs = torch.from_numpy(x_u).float().unsqueeze(-1).unsqueeze(-1)  # (L, D, 1, 1)
        mask = torch.from_numpy(mask).float()                                # (L,)
        labels = torch.from_numpy(labels).float()                            # (L, classes)

        # other: nel tuo codice usa other[0][0] come key e other[1][0] come durata sec
        # per fps = outputs.size()[1] / other[1][0]
        # possiamo mettere durata "fittizia" usando T/window_size non ha senso;
        # meglio leggere FPS vero dal cap_detection_handheld_val_index.csv, ma per eval MAP non serve.
        other = [[vid], [1.0]]

        return inputs, mask, labels, other, torch.from_numpy(hm)
        

def mt_collate_fn(batch):
    # batch list di tuple
    # Charades collate probabilmente impila. Qui facciamo stacking semplice.
    inputs = torch.stack([b[0] for b in batch], dim=0)
    mask   = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.stack([b[2] for b in batch], dim=0)
    other  = [b[3] for b in batch]  # lista di other
    hm     = torch.stack([b[4] for b in batch], dim=0)
    return inputs, mask, labels, other, hm