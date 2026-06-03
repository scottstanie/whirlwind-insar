import sys, glob, time, h5py, numpy as np, whirlwind as ww
tau=2*np.pi; wrap=lambda x:((x+np.pi)%tau)-np.pi
frame=sys.argv[1]
h5=glob.glob(f"/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/nisar_gunw/*_{frame}_*.h5")[0]
base="/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
with h5py.File(h5,"r") as h:
    grp=h[base]; pol=sorted(k for k,v in grp.items() if isinstance(v,h5py.Group) and k.upper() not in {"MASK","METADATA"})[0]
    prod=h[f"{base}/{pol}/unwrappedPhase"][()].astype(np.float32); coh=h[f"{base}/{pol}/coherenceMagnitude"][()].astype(np.float32)
    cc_prod=h[f"{base}/{pol}/connectedComponents"][()].astype(np.int32); ma=h[f"{base}/mask"][()] if "mask" in grp else None
mask=(ma!=255)&((ma//100)%10==0) if ma is not None else np.ones(prod.shape,bool); mask&=np.isfinite(prod)&np.isfinite(coh)
ig=np.exp(1j*np.where(mask,wrap(prod),0.0)).astype(np.complex64); coh_in=np.where(mask,np.clip(np.nan_to_num(coh),0,1),0.0).astype(np.float32)
t=time.time()
unw, cc = ww.unwrap(ig, coh_in, 16.0, mask)   # PUBLIC entry, all defaults
dt=time.time()-t
unw=np.asarray(unw); cc=np.asarray(cc)
in_c=mask&np.isfinite(unw)&(cc_prod>0); amb=np.rint((unw[in_c]-prod[in_c])/tau); ccp=cc_prod[in_c]
ok=tot=0
for lab in np.unique(ccp):
    m=ccp==lab; off=np.median(amb[m]); ok+=int((np.abs(amb[m]-off)<0.5).sum()); tot+=int(m.sum())
print(f"{frame}: ww.unwrap() percomp={ok/tot:.3f}  conncomp: dtype={cc.dtype} shape={cc.shape} n_labels={len(np.unique(cc))-1}  ({dt:.0f}s)",flush=True)
