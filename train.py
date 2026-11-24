import argparse,random,time,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader,Subset
from v4.data import build_pairs,PairedDataset
from v4.model import QRMSRNetV4
from v4.losses import total_loss
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--low_dir",default="data/raw/Learning Data/low"); p.add_argument("--high_dir",default="data/raw/Learning Data/high")
 p.add_argument("--epochs",type=int,default=150); p.add_argument("--batch_size",type=int,default=8); p.add_argument("--lr",type=float,default=2e-4)
 p.add_argument("--base",type=int,default=48); p.add_argument("--seed",type=int,default=42); p.add_argument("--val_ratio",type=float,default=.2)
 p.add_argument("--workers",type=int,default=4); p.add_argument("--output",default="outputs/qr_v4")
 p.add_argument("--edge_weight",type=float,default=.10); p.add_argument("--binary_weight",type=float,default=.05)
 a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 pairs=build_pairs(a.low_dir,a.high_dir); ids=np.random.default_rng(a.seed).permutation(len(pairs))
 nv=max(1,round(len(ids)*a.val_ratio)); tr,va=ids[nv:].tolist(),ids[:nv].tolist()
 ds_train=PairedDataset(pairs,True); ds_val=PairedDataset(pairs,False)
 tl=DataLoader(Subset(ds_train,tr),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=="cuda")
 vl=DataLoader(Subset(ds_val,va),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=="cuda")
 m=QRMSRNetV4(a.base).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
 sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=6)
 out=Path(a.output); (out/"checkpoints").mkdir(parents=True,exist_ok=True); best=-1; hist=[]
 print(f"device={dev} train={len(tr)} val={len(va)} params={sum(p.numel() for p in m.parameters()):,}")
 for ep in range(1,a.epochs+1):
  start=time.time()
  for training,loader in [(True,tl),(False,vl)]:
   m.train(training); losses=[]; ps=[]
   for x,y,_ in loader:
    x,y=x.to(dev),y.to(dev)
    with torch.set_grad_enabled(training):
     o=m(x); loss,parts=total_loss(o,y,a.edge_weight,a.binary_weight)
     if training:
      opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    mse=(o["image"].detach()-y).square().flatten(1).mean(1).clamp_min(1e-12)
    ps.extend((10*torch.log10(1/mse)).cpu().tolist()); losses.append(loss.item())
   if training: trloss=float(np.mean(losses)); trps=float(np.mean(ps))
   else: vloss=float(np.mean(losses)); vps=float(np.mean(ps))
  sch.step(vps); row={"epoch":ep,"train_loss":trloss,"train_psnr":trps,"val_loss":vloss,"val_psnr":vps,"lr":opt.param_groups[0]["lr"],"seconds":time.time()-start}
  hist.append(row); print(f"Epoch {ep}/{a.epochs} trainPSNR={trps:.4f} valPSNR={vps:.4f} loss={vloss:.6f}")
  state={"model":m.state_dict(),"args":vars(a),"epoch":ep,"best_val_psnr":max(best,vps)}
  torch.save(state,out/"checkpoints/last.pth")
  if vps>best: best=vps; torch.save(state,out/"checkpoints/best.pth"); print("[BEST]",best)
  (out/"history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")
 print("Best val PSNR:",best)
if __name__=="__main__": main()
