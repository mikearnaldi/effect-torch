"""CPU lifecycle proof for the generated shared-RNG overlay; pass its directory."""
import ast, sys, json
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, sys.argv[1])
import shared_rng

def extract(path,name,namespace):
 tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
 node.decorator_list=[];module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
 exec(compile(module,str(path),'exec'),namespace);return namespace[name]
source=Path(sys.argv[1])/'diffusion_gemma.py'
ns={'torch':torch,'np':np,'shared_rng':shared_rng}
States=extract(source,'DiffusionGemmaRequestStates',ns)
s=States(3,2,4,48,torch.device('cpu'),3,1)
s.init_canvas(np.array([0,2]));s.set_random_seed(0,7);s.add_request(0)
expected=shared_rng.CanvasStream(7); initial=expected.canvas(2,4)
assert s.canvas[0].tolist()==initial
s.init_canvas(np.array([0]));assert s.canvas[0].tolist()==initial
assert s._canvas_streams[0].draws==1
assert s.canvas_noise([0]).tolist()==[expected.canvas(2,4)]
s.set_random_seed(0,7);assert s.rng_draws[0]==0;assert s._initial_canvases[0]==initial
state=extract(Path(__file__).with_name('vllm-policy-cpu-test.py'),'state',{'torch':torch})
observed=[]
def uniform(template,seeds,draws):
 observed.append(draws.tolist())
 result=torch.empty_like(template)
 for r in range(template.shape[0]):
  result[r].view(-1).copy_(torch.tensor([shared_rng.uniform_f32(int(seeds[r]),int(draws[r]),i) for i in range(result[r].numel())]))
 return result
shared_rng.uniform_like=uniform
sampler=extract(source,'_compiled_sample_step',ns)
x=state(1); x.update(rng_seeds=torch.tensor([7,0,9]),rng_draws=torch.zeros(3,dtype=torch.int64),canvas_noise=torch.tensor([[1,2],[2,3]]))
sampler(**x);assert x['rng_draws'].tolist()==[1,0,1]
x['is_encoder_phase'][x['decode_slots']]=True
sampler(**x);assert x['rng_draws'].tolist()==[1,0,1]
x['is_encoder_phase'][x['decode_slots']]=False
sampler(**x);assert x['rng_draws'].tolist()==[2,0,2]
assert observed==[[0,0],[1,1],[1,1]]
print(json.dumps({'initialCanvasCached':True,'realSeedReset':True,'warmupWithoutAddRequest':True,'denoiseOnlyUniformDraws':observed,'status':'passed'}))
