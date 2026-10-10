"""Read-only source images; disk-backed inventory, incremental CSV, verified copies."""
import argparse,csv,fcntl,hashlib,io,json,math,os,re,shutil,sqlite3,sys,time,uuid
from pathlib import Path
from contextlib import contextmanager
from model_adapter import describe,load_pipeline,probabilities,sha256
# [REDACTED for publication: HPC username removed. Set SCRATCH_ROOT to the data root.]
# BASE=Path('/scratch/<REDACTED_USER>')
BASE=Path(os.environ.get('SCRATCH_ROOT','/path/to/scratch'))
DEFAULT_SOURCES={'original':BASE/'mycorrhisee_tiling_output','rf060_recovered':BASE/'mycorrhisee_rf060_recovery/data'}
DEFAULT_OUTPUT=BASE/'mycorrhisee_swinb_classified'
TILE_DIR='patchsize1152_tilesize576_200microns'
PATTERN=re.compile(r'^(?:yield_rf060__)?(?P<scan_id>.+)_(?P<scan_region>ScanRegion\d+)_half_?(?P<half>\d+)_patchsize(?P<patch_size>\d+)_patch_?(?P<patch>\d+)_tile(?P<tile_size>\d+)_tile_?(?P<tile_number>\d+)\.png$',re.I)
CLASSES=['good','yellow','blurry','opaque']
FIELDS=['original_file_path','original_filename','predicted_class']+[c+'_probability' for c in CLASSES]+['confidence','second_highest_class','second_highest_probability','probability_margin','classified_copy_path','processing_status','source_dataset','source_relative_path','collection','treatment_or_timepoint','block','accession','scan_id','scan_region','half','patch_size','patch','tile_size','tile_number','rf_value','rf_cutoff','source_sha256','error']

def related(a,b):return a==b or a in b.parents or b in a.parents

def safe_roots(sources,output):
 roots={k:Path(v).resolve(strict=True) for k,v in sources.items()}
 out=Path(output).resolve()
 for k,p in roots.items():
  if not re.fullmatch(r'[A-Za-z0-9_-]+',k) or not p.is_dir():raise ValueError('Invalid source alias/directory')
  if related(p,out):raise ValueError(f'Unsafe source/output relationship: {p} <> {out}')
 for i,a in enumerate(roots.values()):
  for b in list(roots.values())[i+1:]:
   if related(a,b):raise ValueError('Overlapping source roots')
 return roots,out

class Store:
 def __init__(self,sources,output):
  self.sources,self.out=safe_roots(sources,output)
  self.out.mkdir(parents=True,exist_ok=True)
  for label in CLASSES:self.guard(self.out/label).mkdir(exist_ok=True)
  self.state=self.guard(self.out/'_state');self.state.mkdir(exist_ok=True)
  self.lock=open(self.guard(self.state/'run.lock'),'a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.db=sqlite3.connect(str(self.guard(self.state/'progress.sqlite')),timeout=60)
  self.db.row_factory=sqlite3.Row
  self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
  self.db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
   CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY,source TEXT UNIQUE NOT NULL,alias TEXT NOT NULL,relative TEXT NOT NULL,size INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,sha TEXT NOT NULL,metadata TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',record TEXT,destination TEXT,seen INTEGER NOT NULL DEFAULT 0);
   CREATE INDEX IF NOT EXISTS image_state ON images(state,id);
   CREATE UNIQUE INDEX IF NOT EXISTS image_dest ON images(destination) WHERE destination IS NOT NULL;''')
  self.db.commit();self.skipped=0;self.last='';self.copy_recovered=0
  self.counts=dict(self.db.execute('SELECT state,count(*) FROM images GROUP BY state'))
  roots={k:str(v) for k,v in self.sources.items()}
  previous=self.get('sources')
  if previous is not None and previous!=roots:raise ValueError('Source roots changed from the persisted inventory')
  self.set('sources',roots)
 def close(self):self.db.close();self.lock.close()
 def guard(self,path):
  p=Path(path);resolved=p.resolve()
  if resolved!=self.out and self.out not in resolved.parents:raise ValueError(f'Output path escapes output root: {p}')
  if any(related(resolved,s) for s in self.sources.values()):raise ValueError('Attempt to write into or above a source')
  
  for part in [p]+list(p.parents):
   if part==self.out:break
   if part.is_symlink():raise ValueError(f'Symlink in output path: {part}')
  return p
 def atomic_json(self,path,value):
  path=self.guard(path);tmp=self.guard(path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp'))
  with open(tmp,'x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path) 
 def get(self,key):
  r=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();return json.loads(r[0]) if r else None
 def set(self,key,value):
  self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value)));self.db.commit()
 def record(self,row,record,state,commit=True):
  old=self.db.execute('SELECT state FROM images WHERE id=?',(row['id'],)).fetchone()[0]
  self.counts[old]=self.counts.get(old,0)-1;self.counts[state]=self.counts.get(state,0)+1
  self.db.execute('UPDATE images SET record=?,state=?,destination=? WHERE id=?',(json.dumps(record),state,record.get('classified_copy_path') or None,row['id']))
  if commit:self.db.commit()
 def progress(self,phase):
  counts={k:v for k,v in self.counts.items() if v}
  predicted=sum(counts.get(k,0) for k in ['success','copy_pending','copy_failed'])
  obj=dict(phase=phase,total_images_discovered=sum(counts.values()),images_processed=predicted+counts.get('failed',0),images_successfully_classified=predicted,images_copied=counts.get('success',0),images_skipped_because_already_completed=self.skipped,images_failed=counts.get('failed',0)+counts.get('copy_failed',0),pending=counts.get('pending',0)+counts.get('copy_pending',0),current_last_processed_file=self.last,states=counts,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  self.atomic_json(self.state/'progress.json',obj);print(json.dumps(obj),flush=True)
  return obj
 def error(self,row,error):
  path=self.guard(self.out/'failures.csv');exists=path.exists()
  with open(path,'a',newline='') as f:
   w=csv.writer(f)
   if not exists:w.writerow(['timestamp_utc','original_file_path','error'])
   w.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),row['source'],str(error)]);f.flush();os.fsync(f.fileno())
 def export_csv(self):
  path=self.guard(self.out/'master_predictions.csv');tmp=self.guard(self.state/'master_rebuild.csv')
  with open(tmp,'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
   for row in self.db.execute('SELECT record FROM images WHERE record IS NOT NULL ORDER BY id'):
    w.writerow(json.loads(row[0]))
   f.flush();os.fsync(f.fileno())
  os.replace(tmp,path) 
 def append_csv(self,records):
  if not records:return
  with open(self.guard(self.out/'master_predictions.csv'),'a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=FIELDS);w.writerows(records);f.flush();os.fsync(f.fileno())

def walk_tiles(sources):
 """Discover microscope tile PNGs, excluding diagnostic figures and .tmp images."""
 for alias,root in sources.items():
  for directory,dirs,files in os.walk(root,followlinks=False):

   for name in dirs:
    if (Path(directory)/name).is_symlink():raise ValueError(f'Symlink source directory: {Path(directory)/name}')
   if Path(directory).name!=TILE_DIR:continue
   rf={};manifest=Path(directory).parent/'tiles.csv'
   if alias=='rf060_recovered' and manifest.is_file():
    with open(manifest,newline='') as f:
     for r in csv.DictReader(f):rf[r['filename']]=r.get('root_fraction','')
   for filename in sorted(files):
    if not PATTERN.match(filename):continue
    p=Path(directory)/filename
    if p.is_symlink():raise ValueError(f'Symlink source tile: {p}')
    resolved=p.resolve(strict=True)
    if root not in resolved.parents:raise ValueError(f'Source escaped root: {p}')
    yield alias,root,p,rf.get(filename,'')

def metadata(alias,root,p,rf):
 relative=p.relative_to(root);parts=relative.parts;m=PATTERN.match(p.name)
 result={k:'' for k in FIELDS};result.update(m.groupdict())
 result.update(original_file_path=str(p),original_filename=p.name,source_dataset=alias,source_relative_path=str(relative),collection=parts[0],treatment_or_timepoint=parts[1] if parts[0].startswith('GWAS') else '',block=next((v for v in parts if re.fullmatch(r'Block\d+',v)),''),accession=parts[-4] if len(parts)>=4 and parts[0].startswith('GWAS') else '',rf_value=rf,rf_cutoff=.60 if alias=='rf060_recovered' else .75)
 return result

def discover(store):
 if store.get('inventory_complete'):return
 store.progress('discovering');count=0
 for alias,root,p,rf in walk_tiles(store.sources):
  st=p.stat();digest=sha256(p)
  if (p.stat().st_size,p.stat().st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise RuntimeError(f'Source changed during inventory: {p}')
  rec=metadata(alias,root,p,rf);rec['source_sha256']=digest
  previous=store.db.execute('SELECT size,mtime_ns,sha FROM images WHERE source=?',(str(p),)).fetchone()
  if previous and tuple(previous)!=(st.st_size,st.st_mtime_ns,digest):raise RuntimeError(f'Source changed since inventory: {p}')
  insert=store.db.execute('INSERT OR IGNORE INTO images(source,alias,relative,size,mtime_ns,sha,metadata) VALUES (?,?,?,?,?,?,?)',(str(p),alias,str(p.relative_to(root)),st.st_size,st.st_mtime_ns,digest,json.dumps(rec)))
  if insert.rowcount:store.counts['pending']=store.counts.get('pending',0)+1
  count+=1;store.last=str(p)
  if count%1000==0:store.db.commit();store.progress('discovering')
 store.db.commit()

 audit_sources(store);store.set('inventory_complete',True)
 store.progress('inventory_ready')

def check_source(row):
 p=Path(row['source']);st=p.stat()
 if (st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):raise RuntimeError(f'Source metadata changed: {p}')

def valid_record(rec,names):
 if rec.get('predicted_class') not in names or not rec.get('original_file_path'):return False
 try:
  probs=[float(rec[c+'_probability']) for c in names]
  if any(not math.isfinite(v) or not 0<=v<=1 for v in probs) or not math.isclose(sum(probs),1,abs_tol=1e-5):return False
 
  order=sorted(range(4),key=lambda i:(-probs[i],i));top,second=order[:2]
  return rec['predicted_class']==names[top] and rec['second_highest_class']==names[second] and math.isclose(float(rec['confidence']),probs[top],abs_tol=1e-6) and math.isclose(float(rec['second_highest_probability']),probs[second],abs_tol=1e-6) and math.isclose(float(rec['probability_margin']),probs[top]-probs[second],abs_tol=1e-6)
 except (ValueError,TypeError,KeyError):return False

def new_prediction(row,probs,names,store,commit=True):
 rec=json.loads(row['metadata']);order=sorted(range(4),key=lambda i:(-probs[i],i));top,second=order[:2]
 rec.update({name+'_probability':float(p) for name,p in zip(names,probs)})
 rec.update(predicted_class=names[top],confidence=float(probs[top]),second_highest_class=names[second],second_highest_probability=float(probs[second]),probability_margin=float(probs[top]-probs[second]),processing_status='copy_pending')
 dest=store.guard(store.out/names[top]/row['alias']/row['relative']);rec['classified_copy_path']=str(dest)
 if not valid_record(rec,names):raise ValueError('Invalid prediction record')
 if dest.exists():raise FileExistsError(f'Unverified destination collision; never overwritten: {dest}')
 
 store.record(row,rec,'copy_pending',commit=commit);return rec

def copy_record(store,row,rec,commit=True):
 check_source(row);dest=store.guard(rec['classified_copy_path'])
 expected=store.out/rec['predicted_class']/row['alias']/row['relative']
 if dest!=expected:raise ValueError('Destination does not match predicted class/source identity')
 if dest.exists():
  prior=store.db.execute('SELECT source,sha,record FROM images WHERE destination=?',(str(dest),)).fetchone()
  if not prior or prior['source']!=row['source'] or prior['sha']!=row['sha'] or sha256(dest)!=row['sha']:raise FileExistsError(f'Unverified destination collision: {dest}')
 else:
  dest.parent.mkdir(parents=True,exist_ok=True);store.guard(dest)
  staging=store.guard(store.state/'staging');staging.mkdir(exist_ok=True)
  tmp=store.guard(staging/(str(row['id'])+'_'+uuid.uuid4().hex+'.part'))

  shutil.copy2(row['source'],tmp,follow_symlinks=False)
  if sha256(tmp)!=row['sha']:raise RuntimeError('Copy differs from inventoried source')
  check_source(row)
  with open(tmp,'rb') as f:os.fsync(f.fileno())

  os.link(tmp,dest,follow_symlinks=False)
  store.guard(tmp).unlink() 
 rec['processing_status']='success';rec['error']='';store.record(row,rec,'success',commit=commit);return rec

def reconcile(store,names,retry_failures=False):
 """Rebuild the CSV from validated durable records before skipping any completed row."""
 store.export_csv() 
 for row in store.db.execute("SELECT * FROM images WHERE state!='pending' ORDER BY id"):
  store.last=row['source'];check_source(row);rec=json.loads(row['record']) if row['record'] else {}
  if row['state']=='failed':
   if retry_failures:
    store.db.execute("UPDATE images SET state='pending',record=NULL,destination=NULL WHERE id=?",(row['id'],))
    store.counts['failed']-=1;store.counts['pending']=store.counts.get('pending',0)+1
   continue
  if not valid_record(rec,names) or rec.get('source_sha256')!=row['sha'] or rec.get('original_file_path')!=row['source']:
   raise RuntimeError(f'Invalid persisted prediction requires review: {row["source"]}')
  if row['state']=='copy_failed' and not retry_failures:continue
  try:
   dest=store.guard(rec['classified_copy_path'])
   if row['state']=='success' and dest.is_file() and sha256(dest)==row['sha']:
    expected=store.out/rec['predicted_class']/row['alias']/row['relative']
    if dest!=expected:raise ValueError('Persisted destination mismatch')
   else:copy_record(store,row,rec)
   store.skipped+=1
  except Exception as e:
   rec.update(processing_status='copy_failed',error=str(e));store.record(row,rec,'copy_failed');store.error(row,e)
  if (store.skipped+1)%1000==0:store.progress('verifying_resume')
 store.db.commit();store.export_csv()

def audit_sources(store):
 store.db.execute('UPDATE images SET seen=0');store.db.commit();count=0
 for alias,root,p,rf in walk_tiles(store.sources):
  row=store.db.execute('SELECT * FROM images WHERE source=?',(str(p),)).fetchone()
  if row is None:raise RuntimeError(f'New source tile appeared: {p}')
  check_source(row)
  if sha256(p)!=row['sha']:raise RuntimeError(f'Source content changed: {p}')
  store.db.execute('UPDATE images SET seen=1 WHERE id=?',(row['id'],));count+=1;store.last=str(p)
  if count%1000==0:store.db.commit();store.progress('auditing_sources')
 store.db.commit();missing=store.db.execute('SELECT count(*) FROM images WHERE seen=0').fetchone()[0]
 if missing:raise RuntimeError(f'{missing} inventoried source tiles are missing')
 return count

class WindowDataset:
 def __init__(self,rows,transform):self.rows=rows;self.transform=transform
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  from PIL import Image
  row=self.rows[index]
  try:
   check_source(row)
   with open(row['source'],'rb') as f:raw=f.read()
   if hashlib.sha256(raw).hexdigest()!=row['sha']:raise RuntimeError('Source content changed')
   with Image.open(io.BytesIO(raw)) as img:tensor=self.transform(img.convert('RGB'))
   return row,tensor,None
  except Exception as e:return row,None,repr(e)

def collate(items):
 import torch
 good=[x for x in items if x[1] is not None];bad=[(r,e) for r,t,e in items if e is not None]
 return ([x[0] for x in good],torch.stack([x[1] for x in good]) if good else None,bad)

def infer(store,desc,batch_size,workers,limit=0,retry_failures=False,defer_finalize=False):
 from torch.utils.data import DataLoader
 names=desc['class_names'];reconcile(store,names,retry_failures);store.progress('loading_model')
 model,transform=load_pipeline(desc);processed=0;lastid=0
 while True:
  n=min(4096,limit-processed) if limit else 4096
  if n<=0:break
  rows=[dict(r) for r in store.db.execute("SELECT * FROM images WHERE state='pending' AND id>? ORDER BY id LIMIT ?",(lastid,n))]
  if not rows:break
  loader=DataLoader(WindowDataset(rows,transform),batch_size=batch_size,num_workers=workers,shuffle=False,pin_memory=True,collate_fn=collate,**({'multiprocessing_context':'spawn','prefetch_factor':2} if workers else {}))
  for good,images,bad in loader:
   output=[]
   for row,error in bad:
    rec=json.loads(row['metadata']);rec.update(processing_status='failed',error=error);store.record(row,rec,'failed');store.error(row,error);output.append(rec);store.last=row['source']
   
   store.append_csv(output);output=[]
   if good:
    probs=probabilities(model,images) # Fatal GPU errors stop safely
    planned=[]
    for row,ps in zip(good,probs):
     try:planned.append((row,new_prediction(row,ps,names,store,commit=False)))
     except Exception as e:
      rec=json.loads(row['metadata']);rec.update(processing_status='failed',error=str(e));store.record(row,rec,'failed',commit=False);store.error(row,e);output.append(rec)
    store.db.commit() 
    for row,rec in planned:
     try:rec=copy_record(store,row,rec,commit=False)
     except Exception as e:
      rec.update(processing_status='copy_failed',error=str(e));store.record(row,rec,'copy_failed',commit=False);store.error(row,e)
     output.append(rec);store.last=row['source']
    store.db.commit()
   store.append_csv(output);processed+=len(good)+len(bad);store.progress('inference')
  lastid=rows[-1]['id']
 store.export_csv();store.progress('inference_stopped' if limit else 'inference_finished')
 if not limit and not defer_finalize:finalize(store,names)

def finalize(store,names):
 """Stream all records and copies, then recheck source identities and SHA256."""
 import numpy as np
 bins=np.linspace(0,1,21);hist={c:[0]*20 for c in ['overall']+names};counts={c:0 for c in names};copy_counts={c:0 for c in names};confidence_sums={c:0. for c in ['overall']+names}
 unresolved=store.db.execute("SELECT count(*) FROM images WHERE state IN ('pending','copy_pending')").fetchone()[0]
 if unresolved:raise RuntimeError(f'{unresolved} images still pending; resume inference first')
 store.export_csv();csv_count=0;success=0
 with open(store.out/'master_predictions.csv',newline='') as f:
  reader=csv.DictReader(f)
  for row,rec in zip(store.db.execute('SELECT * FROM images ORDER BY id'),reader,strict=True):
   csv_count+=1
   if rec!= {k:str(v) for k,v in json.loads(row['record']).items()}:raise RuntimeError('CSV and durable record disagree')
   if rec['original_file_path']!=row['source']:raise RuntimeError('Source path mismatch')
   if row['state'] in ('success','copy_failed'):
    if not valid_record(rec,names):raise RuntimeError('Invalid CSV probabilities or class ordering')
    c=rec['predicted_class'];v=float(rec['confidence']);counts[c]+=1
    if row['state']=='success':
     p=store.guard(rec['classified_copy_path'])
     if not p.is_file() or sha256(p)!=row['sha']:raise RuntimeError(f'Classified copy missing or altered: {p}')
     success+=1;copy_counts[c]+=1
    b=min(19,int(v*20))
    for group in ['overall',c]:hist[group][b]+=1;confidence_sums[group]+=v

 observed=0
 for c in names:
  folder=store.guard(store.out/c)
  if not folder.exists():continue
  for directory,dirs,files in os.walk(folder,followlinks=False):
   for dirname in dirs:store.guard(Path(directory)/dirname)
   for filename in files:
    p=store.guard(Path(directory)/filename)
    r=store.db.execute("SELECT id FROM images WHERE destination=? AND state='success'",(str(p),)).fetchone()
    if r is None:raise RuntimeError(f'Unregistered classified file: {p}')
    observed+=1
 assert observed==success==sum(copy_counts.values())
 predictions=sum(counts.values())
 assert predictions==store.db.execute("SELECT count(*) FROM images WHERE state IN ('success','copy_failed')").fetchone()[0]
 source_count=audit_sources(store)
 assert source_count==csv_count
 failed=source_count-success
 summary=dict(status='complete' if not failed else 'finished_with_failures',total_tiles_processed=source_count,total_successful_predictions=predictions,total_successful_copies=success,total_failures=failed,class_counts=counts,class_counts_successfully_copied=copy_counts,class_percentages={c:100*n/predictions if predictions else 0 for c,n in counts.items()},confidence_histogram_bins=bins.tolist(),confidence_histograms=hist,mean_confidence={c:confidence_sums[c]/(predictions if c=='overall' else counts[c]) if (predictions if c=='overall' else counts[c]) else None for c in ['overall']+names},source_file_set_and_bytes_unchanged=True,csv_and_copies_reconciled=True,confidence_scope='all successful predictions, including any copy failures')
 store.atomic_json(store.out/'summary.json',summary);store.progress(summary['status'])

 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,5));ax.bar(names,[counts[c] for c in names]);ax.set(ylabel='Classified tiles',title='Swin-B predicted classes');fig.tight_layout();fig.savefig(store.guard(store.out/'class_distribution.png'),dpi=180);plt.close(fig)
 fig,axs=plt.subplots(3,2,figsize=(10,10))
 for ax,c in zip(axs.flat,['overall']+names):ax.bar(bins[:-1],hist[c],width=.05,align='edge');ax.set(title=c,xlabel='Confidence',ylabel='Tiles',xlim=(0,1))
 axs.flat[-1].axis('off');fig.tight_layout();fig.savefig(store.guard(store.out/'confidence_distributions.png'),dpi=180);plt.close(fig)
 print(json.dumps(summary,indent=2),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','prepare','run','finalize']);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--source',action='append',help='alias=/absolute/source; repeat for multiple roots');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--retry-failures',action='store_true');p.add_argument('--defer-finalize',action='store_true');a=p.parse_args()
 if a.batch_size<1 or a.workers<0 or a.limit<0:p.error('Invalid batch size, worker count, or limit')
 sources=dict(x.split('=',1) for x in a.source) if a.source else DEFAULT_SOURCES
 if a.command=='preflight':
  roots,out=safe_roots(sources,a.output);d=describe();print(json.dumps(d,indent=2));print('Sources:',roots,'Output:',out);return
 store=Store(sources,a.output)
 try:
  desc=describe();previous=store.get('model_provenance')
  if previous is not None and previous!=desc:raise RuntimeError('Model/code/config/mapping/environment changed since this run; use a different output directory')
  store.set('model_provenance',desc);store.atomic_json(store.state/'model_provenance.json',desc)
  if a.command=='prepare':discover(store);store.export_csv()
  elif a.command=='run':
   if not store.get('inventory_complete'):raise RuntimeError('Run prepare first on a CPU compute node')
   infer(store,desc,a.batch_size,a.workers,a.limit,a.retry_failures,a.defer_finalize)
  else:finalize(store,desc['class_names'])
 except Exception as e:
  store.db.rollback();store.counts=dict(store.db.execute('SELECT state,count(*) FROM images GROUP BY state'))
  store.atomic_json(store.state/'run_error.json',dict(error=repr(e),last_source=store.last));store.progress('stopped_with_error');raise
 finally:store.close()
if __name__=='__main__':main()
