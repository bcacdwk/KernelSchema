"""Execute fixed reference on explicit CPU tensors and model native core.contiguous() writeback."""
from pathlib import Path
import importlib.util,json
import torch
root=Path('kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
spec=importlib.util.spec_from_file_location('gdn_fixed_reference',root/'problem/reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
q=torch.zeros((1,1,2,128),dtype=torch.bfloat16,device='cpu');v=torch.zeros((1,1,4,128),dtype=torch.bfloat16,device='cpu')
s=torch.zeros((1,4,128,128),dtype=torch.bfloat16,device='cpu');g=torch.zeros((1,1,4),device='cpu');cu=torch.tensor([0,1],dtype=torch.int32,device='cpu')
back=torch.full((4,4,256),21,dtype=torch.bfloat16,device='cpu');core=back[:,:,::2]
assert not core.is_contiguous() and core.view(-1).stride()==(2,)
out,final=ref.Model().forward(q,q,v,g,g,s,True,cu_seqlens=cu,chunk_indices=None,chunk_offsets=None,use_qk_l2norm_in_kernel=False,core_attn_out=core)
assert out.shape==(1,1,4,128) and final.dtype==torch.float32
assert torch.all(core[0]==0) and torch.all(core[1:]==21) and torch.all(back[:,:,1::2]==21)
back.fill_(21);native_core=core.contiguous();native_core.view(-1)[:out.numel()].copy_(out.reshape(-1))
assert torch.all(core==21) and torch.all(native_core[0]==0)
out2,final2=ref.Model().forward(q,q,v,g,g,s,False,cu_seqlens=None)
assert final2 is None and torch.equal(out,out2)
result={'scope':'Fixed reference executes on CPU; candidate is not imported/compiled. Its core.contiguous() dataflow is reproduced on CPU, not its CUDA kernel.','reference_tensor_cu_none_ci_co':'accepted and produced correct zero outputs','reference_none_cu_default_normalization':'accepted; output_final_state=False returns None','reference_noncontiguous_core':'view-compatible stride-2 prefix updated, tail and odd-lane backing guards untouched','native_core_contiguous_dataflow':'writes detached contiguous copy; caller storage remains unchanged; source has no copyback','devices':['cpu'],'passed':True}
p=Path(__file__).with_suffix('.json');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
