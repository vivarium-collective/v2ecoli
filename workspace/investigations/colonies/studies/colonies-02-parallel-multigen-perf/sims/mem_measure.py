import psutil, gc, csv, os
import numpy as np
from v2ecoli.colony import make_colony
rss=lambda: psutil.Process().memory_info().rss/1048576
def npmb():
    return sum(o.nbytes for o in gc.get_objects() if isinstance(o,np.ndarray))/1048576
c=make_colony(n_cells=1, cache_dir="out/cache", seed=0,
              jitter_per_second=1e-4, init_mass=200.0, emit_cells=False)
c.run(1.0)
rows=[]
def sample(t):
    n=len(c.state['cells']); r=rss(); m=npmb()
    rows.append({'tick':t,'n_cells':n,'rss_mb':round(r,1),'numpy_mb':round(m,1),
                 'native_mb':round(r-m,1)})
    print(f"tick {t:4d} n={n} rss={r:.0f} numpy={m:.0f} native={r-m:.0f}", flush=True)
phase=350
t=0; sample(t)
for stage in range(3):          # plateaus at N=1, 2, 4
    for _ in range(phase):
        c.run(1.0); t+=1
        if t%25==0: sample(t)
    for cid in list(c.state['cells']):
        c.state['cells'][cid]['ecoli']['instance']._composite.state['agents']['0']['divide']=True
    c.run(1.0); t+=1; sample(t)
out=os.path.expanduser("~/mem_measure.csv")
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE {out} ({len(rows)} rows)", flush=True); print("DONE", flush=True)
