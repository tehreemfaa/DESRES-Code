"""Import the trained experiment's implementation; never rebuild its transforms/head."""
import csv,hashlib,json,sys
import os  # Added for publication: reads the redacted repository path from the environment.
from pathlib import Path
# [REDACTED for publication: HPC username and private lab repository name removed.
#  Set CLASSIFICATION_REPO to the training repository checkout.]
# REPO=Path('/scratch/<REDACTED_USER>/<REDACTED_LAB_REPOSITORY>')
REPO=Path(os.environ.get('CLASSIFICATION_REPO','/path/to/classification-repo'))
RESULT=REPO/'results/swin_b_progressive_thaw'
sys.path.insert(0,str(REPO))

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def describe(result=RESULT):
 import torch,torchvision,PIL
 from torchvision.datasets.folder import find_classes
 from utils.trainer import build_eval_transforms
 result=Path(result).resolve()
 options=json.loads((result/'config.json').read_text())
 if options['experiment_name']!='swin_b_progressive_thaw' or options['model']['backbone']!='swin_b':raise ValueError('Not the final Swin-B progressive-thaw experiment')
 if options['model']['loss_fn']!='cross_entropy':raise ValueError('Expected saved cross_entropy configuration for softmax')
 mappings=[find_classes(str(Path(options['data']['path'])/split)) for split in ['train','validate','test']]
 if any(m!=mappings[0] for m in mappings):raise ValueError('Class mappings disagree between splits')
 classes,mapping=mappings[0]
 names=[c.lower() for c in classes]
 if set(names)!={'good','yellow','blurry','opaque'}:raise ValueError(mapping)
 with open(result/'test_predictions.csv') as f:header=next(csv.reader(f))
 if [c.removesuffix('_probability') for c in header if c.endswith('_probability')]!=names:raise ValueError('Saved prediction column ordering disagrees with ImageFolder mapping')
 sources=[REPO/'utils/model_utils.py',REPO/'utils/trainer.py',REPO/'utils/analysis.py',Path(__file__).resolve(),Path(__file__).resolve().parent/'classify.py']
 return dict(result_dir=str(result),checkpoint=str(result/'best_model.pth'),checkpoint_sha256=sha256(result/'best_model.pth'),config_sha256=sha256(result/'config.json'),class_names=names,class_to_idx=mapping,eval_transform=repr(build_eval_transforms(options)),options=options,implementation_sha256={str(p):sha256(p) for p in sources},versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,pillow=PIL.__version__),precision='torch.autocast(cuda), matching trainer.evaluate_full',probabilities='softmax via trainer.logits_to_probs',best_epoch=json.loads((result/'results.json').read_text())['best_epoch'])

def load_pipeline(description):
 import torch
 from utils.model_utils import load_model
 from utils.trainer import load_checkpoint,build_eval_transforms
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; submit the supplied Slurm job. No CPU fallback or checkpoint-loader replacement.')
 options=description['options']

 model=load_model(options)
 load_checkpoint(model,Path(description['result_dir']))
 model.to('cuda');model.eval()
 return model,build_eval_transforms(options)

def probabilities(model,images):
 import torch
 from utils.trainer import logits_to_probs
 with torch.inference_mode():
  with torch.autocast('cuda'):
   logits=model(images.to('cuda',non_blocking=True))
  if logits.ndim!=2 or logits.shape[1]!=4:raise ValueError(f'Expected four logits: {logits.shape}')

  probs=logits_to_probs(logits.float(),'cross_entropy')
  if not torch.isfinite(probs).all() or not torch.allclose(probs.sum(1),torch.ones(len(probs),device=probs.device),atol=1e-5):raise ValueError('Invalid softmax probabilities')
 return probs.cpu().tolist()
