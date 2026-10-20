"""Explanatory analysis of fatal-outcome correlates (Section 5.6, Tables 11-12, Figure 3).
Usage: python fatal_outcome_correlates.py data/raw/2014_2025_butembo_road_accidents_data.csv
Requires pandas, numpy, scipy, matplotlib."""
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from scipy.stats import norm, chi2
def wilson(f,n,z=1.96):
    p=f/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h
def tab(d,col,minn=0):
    g=d.groupby(col,observed=True).y.agg(['size','sum']).rename(columns={'size':'n','sum':'f'})
    g['pct']=100*g.f/g.n; lo,hi=wilson(g.f.values,g.n.values); g['lo']=100*lo; g['hi']=100*hi
    g['share']=100*g.f/d.y.sum()
    return g[g.n>=minn].sort_values('pct',ascending=False).round(1)
def logit(X,y,it=50):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p); H=X.T@(X*W[:,None]); g=X.T@(y-p)
        step=np.linalg.solve(H,g); b+=step
        if np.abs(step).max()<1e-10: break
    p=1/(1+np.exp(-X@b)); H=X.T@(X*(p*(1-p))[:,None]); V=np.linalg.inv(H)
    ll=np.sum(y*np.log(p)+(1-y)*np.log(1-p))
    return b,np.sqrt(np.diag(V)),ll

import re, json
from scipy.stats import binomtest, chi2_contingency
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pandas as pd, numpy as np, re
import sys
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'data/raw/2014_2025_butembo_road_accidents_data.csv')
def pdate(s):
    s=str(s).strip()
    for f in ('%d/%m/%Y','%Y-%m-%d'):
        try: return pd.to_datetime(s,format=f)
        except: pass
    return pd.NaT
s=df.date_accident.astype(str).str.replace('2106','2016')
df['date']=s.map(pdate)
print('unparsed',df.date.isna().sum(), (df.date_accident.astype(str).str.contains('2106')).sum())
cols=[c for c in df.columns if c not in('date_accident',)]
d=df.drop_duplicates(subset=cols).copy()
print(len(df),len(d),len(df)-len(d))
d['y']=(d.accident_TYPE.str.strip().str.lower()=='fatal').astype(int)
print(d.y.sum(), d.y.mean())
d['year']=d.date.dt.year
d['hour']=d.Hour.astype(str).str.extract(r'(\d{1,2})')[0].astype(float)


ds=d.DRIVERS_STATE.str.strip()
m={'Speeding':'Speeding','Poor parking':'Poor parking','Mechanical failure':'Mechanical failure','Wrong maneuver':'Wrong manoeuvre',
'Failure to maintain safe distance between Vehicles':'Insufficient headway','Involuntary manslaughter':'Recorded as involuntary manslaughter',
'Reckless driving':'Reckless driving','Drunk driving':'Alcohol-impaired','Drunk driving / Speeding':'Alcohol-impaired',
'Wrong overtaking':'Overtaking/crossing error','Improper overtaking':'Overtaking/crossing error','Wrong crossing':'Overtaking/crossing error','Speeding / Wrong crossing':'Overtaking/crossing error'}
d['dstate']=ds.map(m).fillna('Not recorded')
vt=d.TYPE_VEHICULE.replace({'Large truck':'Truck/large truck','Truck':'Truck/large truck','Bicycle':'Other/unknown','Unknown':'Other/unknown'})
d['veh']=vt
L=d.ACCIDENT_LOCATION.str.lower()
def ltype(s):
    if 'roundabout' in s or 'intersection' in s: return 'Roundabout/intersection'
    if 'president of the republic' in s or 'bld pres rep' in s: return 'President of the Republic St.'
    if s.startswith('rn') or 'butembo-beni' in s or 'butembo–kyondo' in s or 'kyondo road' in s: return 'National/peri-urban road'
    if 'ambiance' in s: return 'Ambiance St.'
    return 'Other streets/neighbourhoods'
d['ltype']=L.map(ltype)
h=d.hour
d['hband']=np.select([h.between(9,17),h.between(6,8),h.between(18,20)],['09:00-17:59','06:00-08:59','18:00-20:59'],'21:00-05:59')
de=d.DESCRIPTION_Of_ACCIDENT.str.lower()
d['party']=np.where(de.isna(),'Description missing',np.where(de.str.contains('pedestrian|passerby|passer-by|child|woman|man ',na=False),'Pedestrian mentioned',np.where(de.str.contains('moto|haoj|senk|tvs|bajaj|motorcycle',na=False)&~d.TYPE_VEHICULE.eq('Motorcycle'),'Motorcycle as other party','Other described')))
d['period']=pd.cut(d.year,[2013,2015,2017,2019,2023,2025],labels=['2014-15','2016-17','2018-19','2020-23','2024-25'])
d['yrs']=pd.to_numeric(d.Number_of_years_in_use,errors='coerce')

d['ds2']=d.dstate.replace({'Not recorded':'Other/not recorded'})
refs={'veh':'Car','ds2':'Speeding','ROAD_CONDITION':'Good','hband':'09:00-17:59','ltype':'Other streets/neighbourhoods','period':'2018-19'}
parts=[pd.Series(1.0,index=d.index,name='const')]
for f,r in refs.items():
    parts.append(pd.get_dummies(d[f].astype(str),prefix=f,prefix_sep='=',dtype=float).drop(columns=f'{f}={r}'))
parts.append(((d.yrs.fillna(d.yrs.median())-5)/5).rename('yrs_per5')); X=pd.concat(parts,axis=1)
b,se,_=logit(X.values,d.y.values); R=pd.DataFrame({'or':np.exp(b),'lo':np.exp(b-1.96*se),'hi':np.exp(b+1.96*se)},index=X.columns)
spec=[('Vehicle type','veh',['Motorcycle','Truck/large truck','Bus','Car','Other/unknown'],{'Truck/large truck':'Truck / large truck','Other/unknown':'Bicycle / unknown'}),
('Recorded driver state','ds2',['Reckless driving','Overtaking/crossing error','Alcohol-impaired','Mechanical failure','Poor parking','Recorded as involuntary manslaughter','Insufficient headway','Wrong manoeuvre','Speeding','Other/not recorded'],{'Overtaking/crossing error':'Overtaking / crossing error','Insufficient headway':'Insufficient headway','Recorded as involuntary manslaughter':'"Involuntary manslaughter"','Other/not recorded':'Not recorded'}),
('Road condition','ROAD_CONDITION',['Bad','Good'],{}),
('Time of day','hband',['06:00-08:59','09:00-17:59','18:00-20:59','21:00-05:59'],{}),
('Location type','ltype',['Roundabout/intersection','President of the Republic St.','Other streets/neighbourhoods','Ambiance St.','National/peri-urban road'],{'Roundabout/intersection':'Roundabout / intersection','Other streets/neighbourhoods':'Other streets / neighbourhoods','National/peri-urban road':'National / peri-urban road'})]
rows=[];bold=[]
for title,f,levels,ren in spec:
    bold.append(len(rows)); rows.append([title,'','','',''])
    t=tab(d,f)
    for L in levels:
        r=t.loc[L]; k=f'{f}={L}'
        o='1 (ref.)' if L==refs[f] else f"{R.loc[k,'or']:.2f} ({R.loc[k,'lo']:.2f}–{R.loc[k,'hi']:.2f})"
        rows.append(['   '+(ren.get(L,L).replace('-','–') if L[0].isdigit() else ren.get(L,L)),f"{int(r.n):,}",f"{int(r.f):,}",f"{r.pct:.1f} ({r.lo:.1f}–{r.hi:.1f})",o])
rows.append(['All unique reports',f"{len(d):,}",f"{d.y.sum():,}","19.8 (19.0–20.7)",'—'])
lo,hi=wilson(d.y.sum(),len(d)); print(lo,hi)
print('yrs',R.loc['yrs_per5'].round(2).to_dict())
d['loc']=d.ACCIDENT_LOCATION.str.lower().str.replace(',','').str.replace(r'\s+',' ',regex=True).str.strip()
lt=tab(d,'loc'); lt['bad']=d.groupby('loc').ROAD_CONDITION.apply(lambda s:100*(s=='Bad').mean())
from scipy.stats import binomtest
names={'president of the republic street above cadeco':'President of the Republic St., above CADECO','ambiance street':'Ambiance Street','njiapanda':'Njiapanda','kangote':'Kangote','soficom roundabout':'SOFICOM Roundabout',
'historic roundabout':'Historic Roundabout','president of the republic street above the fec':'President of the Republic St., above the FEC','cap lusando':'Cap Lusando','sonas roundabout':'SONAS Roundabout','ambiance street and market avenue intersection':'Ambiance St. / Market Ave. intersection'}
L=[];bold2=[]
for title,keys in [('(a) Largest number of fatal reports',list(names)[:5]),('(b) Highest fatal share (≥ 50 reports)',list(names)[5:])]:
    bold2.append(len(L)); L.append([title,'','','','',''])
    for k in keys:
        r=lt.loc[k]; p=binomtest(int(r.f),int(r.n),d.y.mean()).pvalue
        L.append(['   '+names[k],f"{int(r.n):,}",f"{int(r.f)}",f"{r.pct:.1f} ({r.lo:.1f}–{r.hi:.1f})",f"{r.share:.1f}",f"{p:.3f}" if p>=0.001 else '<0.001'])
pd.DataFrame(rows).to_csv('table11.csv',index=False,header=False); pd.DataFrame(L).to_csv('table12.csv',index=False,header=False)
for r in rows: print(r)
for r in L: print(r)


def design(d,facs,cont=True):
    parts=[pd.Series(1.0,index=d.index,name='const')]
    for f,r in facs.items():
        D=pd.get_dummies(d[f].astype(str),prefix=f,prefix_sep='=',dtype=float); D=D.drop(columns=f'{f}={r}'); parts.append(D)
    if cont: parts.append(((d.yrs.fillna(d.yrs.median())-5)/5).rename('yrs_per5'))
    return pd.concat(parts,axis=1)
_,_,ll=logit(X.values,d.y.values)
for f in refs:
    r2={k:v for k,v in refs.items() if k!=f}; X2=design(d,r2); _,_,ll2=logit(X2.values,d.y.values)
    df=X.shape[1]-X2.shape[1]; print(f,'LR',round(2*(ll-ll2),2),df,'p=%.4g'%chi2.sf(2*(ll-ll2),df))
X2=design(d,refs,False); _,_,ll2=logit(X2.values,d.y.values); print('yrs LR',2*(ll-ll2),chi2.sf(2*(ll-ll2),1))
# interaction road x morning
d['bm']=((d.ROAD_CONDITION=='Bad')&(d.hband=='06:00-08:59')).astype(float)
Xi=X.copy(); Xi['bad_x_morning']=d.bm; bi,sei,lli=logit(Xi.values,d.y.values)
print('interaction OR',np.exp(bi[-1]),np.exp(bi[-1]-1.96*sei[-1]),np.exp(bi[-1]+1.96*sei[-1]),'LR p',chi2.sf(2*(lli-ll),1))

print(d['loc'].nunique())
t=tab(d,'loc'); p0=d.y.mean()
t50=t[t.n>=50].copy()
t50['p_binom']=[binomtest(int(f),int(n),p0).pvalue for f,n in zip(t50.f,t50.n)]
t50['bad_share']=d[d['loc'].isin(t50.index)].groupby('loc').ROAD_CONDITION.apply(lambda s:100*(s=='Bad').mean()).round(1)
t50['moto_share']=d[d['loc'].isin(t50.index)].groupby('loc').veh.apply(lambda s:100*(s=='Motorcycle').mean()).round(1)
t50['morning']=d[d['loc'].isin(t50.index)].groupby('loc').hband.apply(lambda s:100*(s=='06:00-08:59').mean()).round(1)
pd.set_option('display.width',250); print(t50.sort_values('pct',ascending=False).round(3).to_string())
ct=pd.crosstab(d.loc[d['loc'].isin(t50.index),'loc'],d.y); c=chi2_contingency(ct); print('chi2',c[0],c[2],c[1], len(t50))

rows=[('Vehicle type (ref. car)',None),('Motorcycle','veh=Motorcycle'),('Truck / large truck','veh=Truck/large truck'),('Bus','veh=Bus'),
('Recorded driver state (ref. speeding)',None),('Reckless driving','ds2=Reckless driving'),('Overtaking / crossing error','ds2=Overtaking/crossing error'),('Alcohol-impaired','ds2=Alcohol-impaired'),('Mechanical failure','ds2=Mechanical failure'),('Poor parking','ds2=Poor parking'),
('Road condition (ref. good)',None),('Bad','ROAD_CONDITION=Bad'),
('Time of day (ref. 09:00–17:59)',None),('06:00–08:59','hband=06:00-08:59'),('18:00–20:59','hband=18:00-20:59'),('21:00–05:59','hband=21:00-05:59'),
('Location type (ref. other streets)',None),('Roundabout / intersection','ltype=Roundabout/intersection'),('President of the Republic St.','ltype=President of the Republic St.'),('Ambiance St.','ltype=Ambiance St.'),('National / peri-urban road','ltype=National/peri-urban road')]
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9})
fig,ax=plt.subplots(figsize=(7.0,6.2),dpi=300)
n=len(rows); ylab=[]
for i,(lab,k) in enumerate(rows):
    y=n-1-i
    if k is None:
        ax.text(0.405,y,lab,fontweight='bold',ha='left',va='center',transform=ax.get_yaxis_transform()) if False else None
        ylab.append((y,lab,True)); continue
    r=R.loc[k]; sig=(r.lo>1)|(r.hi<1)
    ax.plot([r.lo,r.hi],[y,y],color='#222' if sig else '#888',lw=1.4,solid_capstyle='round')
    ax.plot(r['or'],y,'o',ms=6,color='#222' if sig else 'white',mec='#222' if sig else '#888',mew=1.3)
    ax.text(1.02,y,f"{r['or']:.2f} ({r.lo:.2f}–{r.hi:.2f})",va='center',ha='left',fontsize=8,color='#222',transform=ax.get_yaxis_transform())
    ylab.append((y,lab,False))
ax.axvline(1,color='#555',lw=0.8,ls='--')
ax.set_xscale('log'); ax.set_xlim(0.45,4.0); ax.set_xticks([0.5,0.75,1,1.5,2,3]); ax.set_xticklabels(['0.5','0.75','1','1.5','2','3']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.text(1.02,n-0.5,'aOR (95% CI)',fontweight='bold',fontsize=8,va='center',ha='left',transform=ax.get_yaxis_transform())
ax.set_yticks([y for y,_,_ in ylab]); ax.set_yticklabels([l if h else '   '+l for _,l,h in ylab])
for t,(_,_,h) in zip(ax.get_yticklabels(),ylab): t.set_fontweight('bold' if h else 'normal')
ax.set_ylim(-0.7,n-0.3); ax.tick_params(axis='y',length=0)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.grid(axis='x',color='#e5e5e5',lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel('Adjusted odds ratio of a fatal outcome (log scale, 95% CI)')
fig.tight_layout(rect=(0,0,0.86,1)); fig.savefig('fig3_forest.png',dpi=300); print('ok')
