import torch
import torch.nn.functional as F
def sobel(x):
    k=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device=x.device,dtype=x.dtype).view(1,1,3,3)/8
    gx=F.conv2d(x,k,padding=1); gy=F.conv2d(x,k.transpose(-1,-2),padding=1)
    return torch.sqrt(gx.square()+gy.square()+1e-8).clamp(0,1)
def total_loss(o,y,edge_weight=.10,binary_weight=.05,aux1_weight=.15,aux2_weight=.05):
    charbonnier=lambda a,b: torch.sqrt((a-b).square()+1e-6).mean()
    rec=charbonnier(o["image"],y); a1=charbonnier(o["aux1"],y); a2=charbonnier(o["aux2"],y)
    edge=F.l1_loss(torch.sigmoid(o["edge_logits"]),sobel(y))
    binary=F.binary_cross_entropy_with_logits(o["binary_logits"],(y>=.5).float())
    loss=rec+aux1_weight*a1+aux2_weight*a2+edge_weight*edge+binary_weight*binary
    return loss,{"rec":rec.detach(),"aux1":a1.detach(),"aux2":a2.detach(),"edge":edge.detach(),"binary":binary.detach()}
