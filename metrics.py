import numpy as np
from skimage.metrics import structural_similarity
def image_metrics(pred,target):
    p=np.clip(np.asarray(pred,dtype=np.float32),0,1)
    t=np.clip(np.asarray(target,dtype=np.float32),0,1)
    mse=float(np.mean((p-t)**2)); psnr= float(10*np.log10(1.0/max(mse,1e-12)))
    pu=np.uint8(np.round(p*255)); tu=np.uint8(np.round(t*255))
    ssim=float(structural_similarity(tu,pu,data_range=255,gaussian_weights=True,
                 sigma=1.5,win_size=11,use_sample_covariance=False))
    return psnr,ssim
