import torch
import torch.nn as nn
import torch.nn.functional as F

def ng(c):
    for g in (8,4,2,1):
        if c % g == 0: return g
    return 1

class CAB(nn.Module):
    def __init__(self,c):
        super().__init__(); h=max(c//8,4)
        self.body=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),
            nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(ng(c),c))
        self.att=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,h,1),
            nn.GELU(),nn.Conv2d(h,c,1),nn.Sigmoid())
        self.scale=nn.Parameter(torch.tensor(.2))
    def forward(self,x):
        z=self.body(x); return x+self.scale*z*self.att(z)

class Down(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1),nn.GELU(),CAB(b),CAB(b))
    def forward(self,x): return self.net(x)

class Up(nn.Module):
    def __init__(self,a,s,b):
        super().__init__(); self.reduce=nn.Conv2d(a,b,1)
        self.fuse=nn.Sequential(nn.Conv2d(b+s,b,3,padding=1),nn.GELU(),CAB(b),CAB(b))
    def forward(self,x,skip):
        x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        return self.fuse(torch.cat([self.reduce(x),skip],1))

class QRMSRNetV4(nn.Module):
    """V2 encoder-decoder + V1-inspired edge/binary structure supervision and gated fusion."""
    def __init__(self,base=48):
        super().__init__(); self.base=base
        self.head=nn.Sequential(nn.Conv2d(1,base,3,padding=1),nn.GELU())
        self.e1=nn.Sequential(CAB(base),CAB(base))
        self.d1=Down(base,base*2); self.e2=nn.Sequential(CAB(base*2),CAB(base*2))
        self.d2=Down(base*2,base*4); self.mid=nn.Sequential(CAB(base*4),CAB(base*4),CAB(base*4))
        self.u2=Up(base*4,base*2,base*2); self.u1=Up(base*2,base,base)
        self.edge_head=nn.Sequential(nn.Conv2d(base,base//2,3,padding=1),nn.GELU(),nn.Conv2d(base//2,1,3,padding=1))
        self.binary_head=nn.Sequential(nn.Conv2d(base,base//2,3,padding=1),nn.GELU(),nn.Conv2d(base//2,1,3,padding=1))
        self.gate=nn.Sequential(nn.Conv2d(base+2,base,1),nn.Sigmoid())
        self.refine=nn.Sequential(nn.Conv2d(base,base,3,padding=1),nn.GELU(),nn.Conv2d(base,base,3,padding=1))
        self.out=nn.Conv2d(base,1,3,padding=1)
        self.aux1=nn.Conv2d(base,1,3,padding=1); self.aux2=nn.Conv2d(base*2,1,3,padding=1)
    def forward(self,x):
        f1=self.e1(self.head(x)); f2=self.e2(self.d1(f1))
        z=self.mid(self.d2(f2)); d2=self.u2(z,f2); d1=self.u1(d2,f1)
        edge=self.edge_head(d1); binary=self.binary_head(d1)
        s=torch.cat([torch.sigmoid(edge),torch.sigmoid(binary)],1)
        d1=d1+self.refine(d1*(1+self.gate(torch.cat([d1,s],1))))
        return {"image":(x+self.out(d1)).clamp(0,1),
                "aux1":(x+self.aux1(d1)).clamp(0,1),
                "aux2":(x+F.interpolate(self.aux2(d2),size=x.shape[-2:],mode="bilinear",align_corners=False)).clamp(0,1),
                "edge_logits":edge,"binary_logits":binary}
