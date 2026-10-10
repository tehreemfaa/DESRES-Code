import csv,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
import recover
import aicsimageio
import spline_guided_tiling_utils_v2 as utils

class RecoveryTests(unittest.TestCase):
 def test_cutoff_boundary(self):
  self.assertFalse(recover.eligible(.599999));self.assertTrue(recover.eligible(.60));self.assertTrue(recover.eligible(.75))
 def test_additive_skip_and_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/'run';source_root=root/'input';old=root/'old';run.mkdir();source_root.mkdir()
   source=source_root/'scan.czi';source.write_bytes(b'fake scan');st=source.stat()
   recover.write_csv(run/'input_inventory.csv',[dict(abs_input_path=str(source),size_bytes=st.st_size,mtime_ns=st.st_mtime_ns)])
   existing=old/'scan'/recover.SUB;existing.mkdir(parents=True)
   original={}
   for i in range(1,5):
    name=f'scan_ScanRegion0_half_1_patchsize1152_patch_1_tile576_tile_{i}.png';(existing/name).write_bytes(b'original bytes must not change');original[name]=(existing/name).read_bytes()
   block=np.full((1152,1152,3),255,np.uint8)
   for bounds,rf in zip(recover.COORDS,[.65,.80,.59,0.]):
    y1,y2,x1,x2=bounds;tile=np.full((576*576,3),255,np.uint8);tile[:int(rf*len(tile))]=100;block[y1:y2,x1:x2]=tile.reshape(576,576,3)
   class FakeImage:
    scenes=['ScanRegion0']
    def __init__(self,*a):pass
    def set_scene(self,*a):pass
    def get_image_data(self,*a,**k):return np.full((2,2,3),100,np.uint8)
   with patch.multiple(recover,RUN=run,INPUT=source_root,OLD=old),patch.object(aicsimageio,'AICSImage',FakeImage),patch.object(utils,'preprocess_image',return_value=[np.full((2,2,3),100,np.uint8)]),patch.object(cv2,'copyMakeBorder',side_effect=lambda a,*args,**kwargs:a),patch.object(utils,'find_largest_contour_with_area',return_value=(1,np.array([0,1]),np.array([0,1]),1)),patch.object(utils,'find_squares',return_value=([0,1],[0,1],1)),patch.object(utils,'warp_and_crop',return_value=block) as warp:
    recover.recover(0)
    self.assertEqual(warp.call_count,1) # patch 1 already exists; no extraction
    output=run/'data/scan';r=json.loads((output/'complete.json').read_text())
    self.assertEqual((r['candidate_tiles'],r['existing_tiles_skipped'],r['new_tiles_060_075'],r['missing_prior_075_plus']),(8,4,1,1))
    files=list((output/recover.SUB).glob('*.png'));self.assertEqual(len(files),2)
    hashes={f.name:f.read_bytes() for f in files}
    recover.recover(0);self.assertEqual(warp.call_count,1)
    # Resume a scan without a final marker from its scene checkpoint.
    (output/'complete.json').unlink();recover.recover(0);self.assertEqual(warp.call_count,1)
    self.assertEqual(hashes,{f.name:f.read_bytes() for f in files})
    self.assertEqual(original,{f.name:f.read_bytes() for f in existing.glob('*.png')})
if __name__=='__main__':unittest.main()
