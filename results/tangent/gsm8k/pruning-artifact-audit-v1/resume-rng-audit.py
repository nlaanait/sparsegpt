"""CPU check of local calibration/atlas independence from global RNG history."""
from pathlib import Path
from hashlib import sha256
import json,re,tempfile
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from sparsifier.eval.gsm8k_sequential import collect
import sparsifier.eval.gsm8k_sequential as sequential
from sparsifier.prune.factor_atlas import FactorAtlas,projector_diagonals
from sparsifier.prune.gram_dictionary import GramBlockDictionary

class Tokenizer:
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
    def encode(self,text,**kwargs):
        index=int(re.search(r'\d+$',text).group())
        return [1+(index+position)%28 for position in range(63+index%5)]+[4+index%10]

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(32,8);self.anchor=nn.Linear(8,12);self.head=nn.Linear(12,32)
    def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.head(self.anchor(self.embedding(input_ids)))[:,-1:])

def perturb(seed):
    np.random.seed(seed);np.random.normal(size=10000)
    torch.manual_seed(seed);torch.rand(10000)

def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4);torch.manual_seed(1234)
    model=Model().eval().requires_grad_(False);tokenizer=Tokenizer()
    rows=[{'id':str(i),'question':str(i)} for i in range(128)]
    c4=np.random.default_rng(901).integers(1,32,size=(32,128)).tolist()
    positions=[]; original_capture=sequential.capture_batch
    def captured(*args,**kwargs):
        result=original_capture(*args,**kwargs)
        positions.extend(r['positions'].tolist() for r in result)
        return result
    sequential.capture_batch=captured
    try:
        perturb(31);left=collect(model,tokenizer,model.anchor,rows,c4,0);first_positions=list(positions);positions.clear()
        perturb(701);right=collect(model,tokenizer,model.anchor,rows,c4,0)
        assert first_positions==positions and len(positions)==160
    finally:sequential.capture_batch=original_capture
    arrays=lambda values:[v.numpy() if isinstance(v,torch.Tensor) else v for v in values]
    capture_equal=[bool(np.array_equal(a,b)) for a,b in zip(arrays(left),arrays(right))]
    assert all(capture_equal) and left[2].shape==(8192,8)
    outputs=np.random.default_rng(511).normal(size=(512,32))
    def fit():
        atlas=FactorAtlas(8,4,2,4,1,0);atlas.dictionary=GramBlockDictionary(8,4,2,0)
        return atlas.fit(outputs)
    perturb(41);a=fit();perturb(801);b=fit()
    with tempfile.TemporaryDirectory(prefix='gsm8k-resume-rng-') as directory:
        p,q=[Path(directory)/name for name in ('first.npz','second.npz')];a.save(p);b.save(q)
        with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
            assert x.files==y.files
            atlas_equal={key:bool(np.array_equal(x[key],y[key])) for key in x.files}
    assert all(atlas_equal.values())
    pa=projector_diagonals(a.frames(outputs,mode='local',seed=0)[0]);pb=projector_diagonals(b.frames(outputs,mode='local',seed=0)[0])
    assert np.array_equal(pa,pb) and not torch.cuda.is_initialized()
    root=Path.cwd();paths=['src/sparsifier/eval/gsm8k_sequential.py','src/sparsifier/eval/gsm8k_capture.py','src/sparsifier/prune/geometry_features.py','src/sparsifier/prune/factor_atlas.py','src/sparsifier/prune/gram_dictionary.py','src/sparsifier/prune/sparsegpt.py','src/sparsifier/eval/gsm8k_pruning.py']
    result={'scope':'toy_CPU_sampling_and_atlas_RNG_history_check_not_GPU_mask_equivalence','capture_examples':128,'sampled_position_arrays_equal':160,'ambient_tokens':8192,'atlas_fit_samples':512,'atlas_features':32,'seed':0,'global_rng_histories':{'capture':[31,701],'atlas':[41,801]},'capture_arrays_equal':dict(zip(['readout_x','readout_y','ambient_inputs','c4_inputs'],capture_equal)),'all_atlas_saved_arrays_equal':all(atlas_equal.values()),'atlas_arrays_checked':len(atlas_equal),'projector_diagonals_equal':True,'source_sha256':{name:sha256((root/name).read_bytes()).hexdigest() for name in paths},'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'used_cuda':False,'limitation':'Does not prove bitwise equality of model-wide GPU runs across process restarts. Code uses fresh local generators seeded per example,calibration collection,dictionary fit,and chart fit; existing saved checkpoint hashes are separately verified.'}
    path=root/'results/tangent/gsm8k/eos-corrected-v1/audits/local-s60-resume-rng.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
