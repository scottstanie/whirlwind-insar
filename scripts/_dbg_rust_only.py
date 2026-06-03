import sys, glob, h5py, numpy as np, whirlwind as ww
tau = 2*np.pi
wrap = lambda x: ((x+np.pi)%tau)-np.pi
frame = sys.argv[1]
h5 = glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]
base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
with h5py.File(h5,"r") as h:
    grp=h[base]
    pol=sorted(k for k,v in grp.items() if isinstance(v,h5py.Group) and k.upper() not in {"MASK","METADATA"})[0]
    prod=h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32)
    coh=h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    ma=h[f"{base}/mask"][()] if "mask" in grp else None
mask=(ma!=255)&((ma//100)%10==0) if ma is not None else np.ones(prod.shape,bool)
mask&=np.isfinite(prod)&np.isfinite(coh)
ig=np.exp(1j*np.where(mask,wrap(prod),0.0)).astype(np.complex64)
coh_in=np.where(mask,np.clip(np.nan_to_num(coh),0,1),0.0).astype(np.float32)
print(f"{frame}: calling unwrap_linear (DEBUG build, asserts live)...",flush=True)
u=ww._native.unwrap_linear(ig,coh_in,16.0,mask)
print(f"{frame}: unwrap_linear returned OK, no assert fired. finite={np.isfinite(np.asarray(u)[mask]).all()}",flush=True)
