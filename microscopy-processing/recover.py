"""Add missing RF>=0.60 tiles without re-exporting production tiles."""
import argparse,csv,hashlib,json,os,sys,time,fcntl
from pathlib import Path
# [REDACTED for publication: HPC username removed. Set SCRATCH_ROOT to the data root.]
# ROOT=Path('/scratch/<REDACTED_USER>')
ROOT=Path(os.environ.get('SCRATCH_ROOT','/path/to/scratch'))
RUN=ROOT/'mycorrhisee_rf060_recovery'
INPUT=ROOT/'mycorrhisee/MycorrhiSEE'
OLD=ROOT/'mycorrhisee_tiling_output'
SUB='patchsize1152_tilesize576_200microns'
COORDS=[(0,576,0,576),(0,576,576,1152),(576,1152,0,576),(576,1152,576,1152)]

def write_json(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)
def write_csv(p,rows,fields=None):
    if not rows and fields is None:return
    tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def eligible(rf):return rf>=.60

def recover(index,max_scenes=0,smoke=False):
    import cv2,numpy as np
    from aicsimageio import AICSImage
    from spline_guided_tiling_utils_v2 import preprocess_image,find_largest_contour_with_area,find_squares,warp_and_crop
    cv2.setNumThreads(2)
    with open(RUN/'input_inventory.csv') as inventory:
        records=list(csv.DictReader(inventory))
    rec=records[index]
    source=Path(rec['abs_input_path']);rel=source.relative_to(INPUT)
    stat=source.stat()
    assert stat.st_size==int(rec['size_bytes']) and stat.st_mtime_ns==int(rec['mtime_ns']), 'Input changed after inventory'
    out=RUN/('smoke' if smoke else 'data')/rel.with_suffix('');out.mkdir(parents=True,exist_ok=True)
    with open(out/'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():print('Already complete',rel);return
        dest=out/SUB;dest.mkdir(exist_ok=True)
        olddir=OLD/rel.with_suffix('')/SUB
        existing={p.name for p in olddir.glob('*.png')}
        recovered=[];scene_rows=[];issues=[];start=time.time()
        reader=AICSImage(str(source));scenes=reader.scenes[:max_scenes or None]
        print('START',index,rel,'scenes',len(scenes),flush=True)
        for scene in scenes:
            checkpoint=out/f'{scene}.json'
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text())
                if all((dest/t['filename']).is_file() for t in saved['tiles']):
                    recovered.extend(saved['tiles']);scene_rows.append(saved['summary']);issues.extend(saved['issues']);continue
            local=[];warnings=[]
            s=dict(scene=scene,status='success',candidate_tiles=0,existing_tiles_skipped=0,rf_below_060=0,recovered_060_075=0,missing_prior_075_plus=0,patches_not_warped=0)
            try:
                reader.set_scene(scene);im=reader.get_image_data('YXS',T=0,C=0,Z=0)
                if im.ndim!=3 or im.shape[-1]!=3:s['status']='non_rgb_reference'
                else:
                    im[(im==[0,0,0]).all(axis=-1)]=255
                    halves=preprocess_image(im,threshold=175,gray=False,rotate=True,half=True,half_rotation=False)
                    for hi,half in enumerate(halves,1):
                        if half is None:continue
                        half[(half==[0,0,0]).all(axis=-1)]=255
                        padded=cv2.copyMakeBorder(half,2304,2304,2304,2304,cv2.BORDER_CONSTANT,value=[255]*3)
                        gray=cv2.cvtColor(padded,cv2.COLOR_BGR2GRAY)
                        contour,x,y,_=find_largest_contour_with_area(gray,175)
                        if contour is None:continue
                        try:xs,ys,_=find_squares(x,y,x.min(),x.max(),side_length=1152,step_size=1152,move=True,ratio=.33)
                        except ValueError as e:
                            warnings.append(dict(scene=scene,half=hi,kind='production_spline_skip',error=str(e)));continue
                        for pi,(xc,yc) in enumerate(zip(xs,ys),1):
                            names=[f'{source.stem}_{scene}_half_{hi}_patchsize1152_patch_{pi}_tile576_tile_{ti}.png' for ti in range(1,5)]
                            if all(name in existing for name in names):
                                s['candidate_tiles']+=4;s['existing_tiles_skipped']+=4;s['patches_not_warped']+=1;continue
                            try:patch=warp_and_crop(padded,xc,yc,side_length=1152)
                            except (ValueError,cv2.error) as e:
                                warnings.append(dict(scene=scene,half=hi,patch=pi,kind='production_warp_skip',error=str(e)));continue
                            for ti,((y1,y2,x1,x2),name) in enumerate(zip(COORDS,names),1):
                                s['candidate_tiles']+=1
                                if name in existing:s['existing_tiles_skipped']+=1;continue
                                tile=patch[y1:y2,x1:x2]
                                rf=float(np.mean(cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)<175))
                                if not eligible(rf):s['rf_below_060']+=1;continue
                                kind='recovered_060_075' if rf<.75 else 'missing_prior_075_plus';s[kind]+=1
                                filename='yield_rf060__'+name;target=dest/filename
                                if target.exists():
                                    previous=cv2.imread(str(target))
                                    if previous is None or not np.array_equal(previous,tile):raise RuntimeError(f'Existing recovery tile differs: {target}')
                                else:
                                    tmp=target.with_suffix('.tmp.png')
                                    if not cv2.imwrite(str(tmp),tile):raise IOError(tmp)
                                    os.replace(tmp,target)
                                local.append(dict(rel_input_path=str(rel),experiment='yield_rf060',recovery_kind=kind,scene=scene,half=hi,patch=pi,tile=ti,root_fraction=rf,original_filename=name,filename=filename,output_path=str(target)))
                write_json(checkpoint,dict(summary=s,tiles=local,issues=warnings))
            except Exception as e:
                s['status']='failed';warnings.append(dict(scene=scene,kind='scene_failure',error=repr(e)))
            recovered.extend(local);scene_rows.append(s);issues.extend(warnings)
            print(scene,s,flush=True)
        candidate=sum(s['candidate_tiles'] for s in scene_rows);old=sum(s['existing_tiles_skipped'] for s in scene_rows)
        added=sum(s['recovered_060_075'] for s in scene_rows);gap=sum(s['missing_prior_075_plus'] for s in scene_rows)
        failures=sum(s['status']=='failed' for s in scene_rows)
        result=dict(index=index,rel_input_path=str(rel),status='failed' if failures else ('complete_with_skips' if issues else 'complete'),scenes_total=len(reader.scenes),scenes_attempted=len(scenes),scenes_failed=failures,non_rgb_scenes=sum(s['status']=='non_rgb_reference' for s in scene_rows),candidate_tiles=candidate,existing_tiles_skipped=old,new_tiles_060_075=added,missing_prior_075_plus=gap,new_tiles_total=added+gap,combined_tiles_in_processed_candidates=old+added+gap,combined_yield=(old+added+gap)/candidate if candidate else None,existing_pngs_in_scan=len(existing),production_skips=sum(i['kind']!='scene_failure' for i in issues),elapsed_seconds=round(time.time()-start,2),rf_cutoff=.60,pixel_threshold=175,scope='smoke' if smoke else 'all_scenes',output_dir=str(out))
        write_csv(out/'tiles.csv',recovered,['rel_input_path','experiment','recovery_kind','scene','half','patch','tile','root_fraction','original_filename','filename','output_path'])
        write_csv(out/'scene_yield.csv',scene_rows);write_json(out/'issues.json',issues);write_json(out/'summary.json',result)
        if failures:raise RuntimeError(f'{failures} failed scenes; rerun resumes successful scene checkpoints')
        write_json(out/'complete.json',result);print('COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--max-scenes',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.max_scenes and not a.smoke:ap.error('Scene limits require --smoke; production always runs all scenes')
    recover(a.index,a.max_scenes,a.smoke)
