import csv,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import classify as c

NAMES=['blurry','good','opaque','yellow']
NAME='Plant_C01_R02_ScanRegion0_half_1_patchsize1152_patch_1_tile576_tile_1.png'

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.sources={}
  for alias in ['original','rf060_recovered']:
   # [REDACTED for publication: study collection, treatment, and block names replaced with generic
   #  ones. The 'GWAS' prefix is kept because classify.metadata() branches on it.]
   src=self.root/alias;p=src/'GWAS_example/TreatmentA/Block1/Plant/Scan'/c.TILE_DIR/NAME;p.parent.mkdir(parents=True)
   Image.new('RGB',(32,32),(123,70,80)).save(p);self.sources[alias]=src
  self.out=self.root/'classified';self.store=c.Store(self.sources,self.out);c.discover(self.store)
 def tearDown(self):self.store.close();self.tmp.cleanup()
 def rows(self):return list(self.store.db.execute('SELECT * FROM images ORDER BY id'))
 def test_path_guards(self):
  with self.assertRaises(ValueError):c.safe_roots(self.sources,self.sources['original']/'output')
  with self.assertRaises(ValueError):c.safe_roots(self.sources,self.root)
  escape=self.out/'escape';escape.symlink_to(self.sources['original'],target_is_directory=True)
  with self.assertRaises(ValueError):self.store.guard(escape/'file.png')
 def test_class_mapping_copy_and_resume(self):
  before={r['source']:c.sha256(r['source']) for r in self.rows()}
  for r in self.rows():
   rec=c.new_prediction(r,[.02,.91,.03,.04],NAMES,self.store);c.copy_record(self.store,r,rec)
   self.assertEqual(rec['predicted_class'],'good');self.assertEqual(rec['second_highest_class'],'yellow');self.assertAlmostEqual(rec['probability_margin'],.87)
  dests=[r['destination'] for r in self.rows()];self.assertNotEqual(*dests)
  self.store.export_csv();(self.out/'master_predictions.csv').write_text('interrupted CSV')
  c.reconcile(self.store,NAMES)
  self.assertEqual(self.store.skipped,2)
  with open(self.out/'master_predictions.csv') as f:self.assertEqual(len(list(csv.DictReader(f))),2)
  self.assertEqual(before,{r['source']:c.sha256(r['source']) for r in self.rows()})
  c.finalize(self.store,NAMES)
  summary=json.loads((self.out/'summary.json').read_text());self.assertEqual(summary['total_successful_copies'],2);self.assertTrue(summary['source_file_set_and_bytes_unchanged'])
 def test_missing_copy_is_repaired_without_model(self):
  r=self.rows()[0];rec=c.new_prediction(r,[.1,.7,.1,.1],NAMES,self.store);c.copy_record(self.store,r,rec)
  Path(rec['classified_copy_path']).unlink() # Test deliberately simulates a missing OUTPUT copy.
  c.reconcile(self.store,NAMES);self.assertTrue(Path(rec['classified_copy_path']).is_file())
 def test_collision_never_overwritten(self):
  r=self.rows()[0];dest=self.out/'good'/r['alias']/r['relative'];dest.parent.mkdir(parents=True);dest.write_bytes(b'belongs to another run')
  with self.assertRaises(FileExistsError):c.new_prediction(r,[.01,.97,.01,.01],NAMES,self.store)
  self.assertEqual(dest.read_bytes(),b'belongs to another run')
 def test_interruption_after_publish(self):
  r=self.rows()[0];rec=c.new_prediction(r,[.1,.7,.1,.1],NAMES,self.store)
  original_record=self.store.record
  def interrupted(row,record,state,**kwargs):
   if state=='success':raise RuntimeError('simulated crash after publication')
   return original_record(row,record,state,**kwargs)
  with patch.object(self.store,'record',side_effect=interrupted):
   with self.assertRaises(RuntimeError):c.copy_record(self.store,r,rec)
  self.assertEqual(self.rows()[0]['state'],'copy_pending')
  c.reconcile(self.store,NAMES);self.assertEqual(self.rows()[0]['state'],'success')
 def test_source_changes_detected(self):
  Path(self.rows()[0]['source']).write_bytes(b'changed by test fixture')
  with self.assertRaises(RuntimeError):c.audit_sources(self.store)
 def test_invalid_probability_and_tie(self):
  r=self.rows()[0]
  with self.assertRaises(ValueError):c.new_prediction(r,[.1,.1,.1,.1],NAMES,self.store)
  rec=c.new_prediction(r,[.5,.5,0,0],NAMES,self.store);self.assertEqual(rec['predicted_class'],'blurry');self.assertEqual(rec['probability_margin'],0)
 def test_unreadable_image_logged_without_placeholder(self):
  r=dict(self.rows()[0]);p=Path(r['source']);p.write_bytes(b'not an image');s=p.stat();r.update(size=s.st_size,mtime_ns=s.st_mtime_ns,sha=c.sha256(p))
  dataset=c.WindowDataset([r],lambda img:img)
  row,tensor,error=dataset[0];self.assertIsNone(tensor);self.assertIn('UnidentifiedImageError',error)
 def test_batched_inference_and_corrupt_input(self):
  from torchvision.transforms import ToTensor
  r=dict(self.rows()[1]);p=Path(r['source']);p.write_bytes(b'bad png fixture');st=p.stat();digest=c.sha256(p)
  meta=json.loads(r['metadata']);meta['source_sha256']=digest
  self.store.db.execute('UPDATE images SET size=?,mtime_ns=?,sha=?,metadata=? WHERE id=?',(st.st_size,st.st_mtime_ns,digest,json.dumps(meta),r['id']));self.store.db.commit()
  with patch.object(c,'load_pipeline',return_value=(object(),ToTensor())),patch.object(c,'probabilities',side_effect=lambda model,images:[[.01,.95,.02,.02] for _ in images]):
   c.infer(self.store,{'class_names':NAMES},1,0)
  summary=json.loads((self.out/'summary.json').read_text())
  self.assertEqual(summary['total_tiles_processed'],2);self.assertEqual(summary['total_successful_predictions'],1);self.assertEqual(summary['total_failures'],1)
  self.assertEqual(summary['class_counts']['good'],1)
  with open(self.out/'master_predictions.csv') as f:records=list(csv.DictReader(f))
  self.assertEqual(len(records),2);self.assertEqual(sorted(r['processing_status'] for r in records),['failed','success'])
  prog=json.loads((self.store.state/'progress.json').read_text());self.assertEqual(prog['images_processed'],2);self.assertEqual(prog['total_images_discovered'],2)
 def test_copy_failures_reconcile_predictions(self):
  from torchvision.transforms import ToTensor
  with patch.object(c,'load_pipeline',return_value=(object(),ToTensor())),patch.object(c,'probabilities',side_effect=lambda model,images:[[.01,.95,.02,.02] for _ in images]),patch.object(c.shutil,'copy2',side_effect=OSError('simulated disk full')):
   c.infer(self.store,{'class_names':NAMES},2,0)
  summary=json.loads((self.out/'summary.json').read_text())
  self.assertEqual(summary['total_successful_predictions'],2);self.assertEqual(sum(summary['class_counts'].values()),2)
  self.assertEqual(summary['total_successful_copies'],0);self.assertEqual(summary['total_failures'],2)
  c.reconcile(self.store,NAMES,retry_failures=True);c.finalize(self.store,NAMES)
  summary=json.loads((self.out/'summary.json').read_text());self.assertEqual(summary['total_successful_copies'],2)
if __name__=='__main__':unittest.main()
