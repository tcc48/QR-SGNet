from pathlib import Path
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset
EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
def list_images(folder):
    p=Path(folder)
    return sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)
def build_pairs(low_dir,high_dir):
    lo,hi=Path(low_dir),Path(high_dir); lows=list_images(lo); highs=list_images(hi)
    rel={p.relative_to(hi).as_posix():p for p in highs}; stems={}
    for p in highs: stems.setdefault(p.stem,[]).append(p)
    used=set(); pairs=[]
    for lp in lows:
        hp=rel.get(lp.relative_to(lo).as_posix())
        if hp is None:
            cand=[p for p in stems.get(lp.stem,[]) if p not in used]
            if len(cand)==1: hp=cand[0]
        if hp is not None and hp not in used: pairs.append((lp,hp)); used.add(hp)
    if not pairs: raise RuntimeError(f"No paired images: {lo} / {hi}")
    print(f"Pairs={len(pairs)} low={len(lows)} high={len(highs)}")
    return pairs
def read_gray(path):
    a=np.asarray(Image.open(path).convert("L"),dtype=np.float32)/255.
    return torch.from_numpy(a).unsqueeze(0)
class PairedDataset(Dataset):
    def __init__(self,pairs,augment=False): self.pairs=pairs; self.augment=augment
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        lp,hp=self.pairs[i]; x,y=read_gray(lp),read_gray(hp)
        if x.shape!=y.shape: raise ValueError(f"Size mismatch: {lp} {hp}")
        if self.augment:
            if torch.rand(())<.5: x,y=x.flip(-1),y.flip(-1)
            if torch.rand(())<.5: x,y=x.flip(-2),y.flip(-2)
            k=int(torch.randint(0,4,()).item())
            if k: x,y=torch.rot90(x,k,(-2,-1)),torch.rot90(y,k,(-2,-1))
        return x,y,str(lp)
