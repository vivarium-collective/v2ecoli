"""Per-chromosome oriC-low DnaA-ATP box COUNTS, parent / daughter1 / daughter2.
Honest: dnaa_box_pool_label==2 = oriC-low; dnaa_box_bound_form==1 = ATP-bound;
dnaa_box_domain_index = chromosome domain. Roles assigned PER FRAME by the number
of oriC-low domains present: 1 domain = the (pre-init) parent oriC; 2 domains = the
two daughter oriCs (sorted by domain id -> daughter1, daughter2). Counts are out of
the per-domain oriC-low site total (~8). init = first frame with 2 oriC-low domains."""
import sys, glob, json, collections, polars as pl
run, out = sys.argv[1], sys.argv[2]; P="listeners__replication_data__"
fs=sorted(glob.glob(f"{run}/**/history/**/*.pq",recursive=True))
cols=["generation","global_time",P+"dnaa_box_pool_label",P+"dnaa_box_domain_index",P+"dnaa_box_bound_form"]
df=pl.scan_parquet(fs,hive_partitioning=True).select(cols).collect().sort(["generation","global_time"])
gens=df["generation"].to_list(); gt=df["global_time"].to_list()
labels=df[P+"dnaa_box_pool_label"].to_list(); doms=df[P+"dnaa_box_domain_index"].to_list(); forms=df[P+"dnaa_box_bound_form"].to_list()
LOW,ATP=2,1
def by_domain(i):
    lab,dom,frm=labels[i],doms[i],forms[i]; d=collections.defaultdict(lambda:[0,0])
    for k in range(len(lab)):
        if lab[k]==LOW:
            d[dom[k]][1]+=1
            if frm[k]==ATP: d[dom[k]][0]+=1
    return d
def roles(i):
    d=by_domain(i); ds=sorted(d.keys(),key=lambda x:(-d[x][1],x))  # by #sites desc then id
    if len(d)==0: return (None,None,None,None)
    if len(d)==1: dom=next(iter(d)); return ("parent",d[dom][0],None,None)  # parent atp count
    pair=sorted(ds[:2])  # the two domains with most oriC-low sites, sorted by id
    a,b=pair; return ("daughters",None,d[a][0],d[b][0])
res={}
for g in sorted(set(gens)):
    idxs=[i for i,gg in enumerate(gens) if gg==g]
    init_min=None
    for i in idxs:
        if len(by_domain(i))>=2: init_min=round(gt[i]/60.0,3); break
    n=len(idxs); step=max(1,n//500); t=[];pc=[];c1=[];c2=[]
    for j in range(0,n,step):
        i=idxs[j]; r=roles(i); t.append(round(gt[i]/60.0,3))
        pc.append(r[1]); c1.append(r[2]); c2.append(r[3])
    res[str(g)]={"t_min":t,"init_min":init_min,"parent":pc,"daughter1":c1,"daughter2":c2,"n_sites":8}
json.dump({"run":run.split("/")[-1],"atp_form":ATP,"n_sites":8,"gens":res},open(out,"w"))
# verify
for g in ("2","5","6"):
    gd=res[g]; mx=lambda a:max([x for x in a if x is not None] or [None])
    print(f"gen {g}: init@{gd['init_min']} parent_max={mx(gd['parent'])} d1_max={mx(gd['daughter1'])} d2_max={mx(gd['daughter2'])}")
