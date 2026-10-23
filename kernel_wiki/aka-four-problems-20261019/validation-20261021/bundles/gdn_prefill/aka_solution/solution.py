# SPDX-License-Identifier: BSD-3-Clause
# BF16/M64 GDN expert candidate; see ../README.md and ../provenance/.
_J='compact_cg1_events'
_I='use_static_scale'
_H='static_tma_full_chunks'
_G='use_static_tma_descriptors'
_F='use_m64_rows'
_E='GDNTileScheduler'
_D=.0
_C=False
_B=True
_A=None
from typing import Tuple,Optional
from cutlass.cutlass_dsl import Integer,Int32,min,extract_mlir_values,new_from_mlir_values,dsl_user_op
from cutlass._mlir import ir
import cutlass,cutlass.cute as cute
from cutlass.utils import WorkTileInfo
class GDNTileSchedulerParams:
	def __init__(self,num_seqs:cutlass.Int32,num_q_heads:cutlass.Int32,num_v_heads:cutlass.Int32,is_GQA:bool,is_persistent:bool=_B,single_sequence_direct:bool=_C,total_tiles:Optional[cutlass.Int32]=_A,num_seqs_fdd:cute.FastDivmodDivisor=_A,num_heads_fdd:cute.FastDivmodDivisor=_A,*,loc=_A,ip=_A):
		H=num_heads_fdd;G=num_seqs_fdd;F=total_tiles;E=is_GQA;D=num_v_heads;C=num_q_heads;B=loc;A=num_seqs;self.num_seqs=A;self.num_q_heads=C;self.num_v_heads=D;self.is_GQA=E;self.is_persistent=is_persistent;self.single_sequence_direct=single_sequence_direct;self.num_o_heads=C if E else D
		if F is _A:self.total_tiles=A*self.num_o_heads
		else:self.total_tiles=F
		self.num_seqs_fdd=G;self.num_heads_fdd=H
		if G is _A:self.num_seqs_fdd=cute.fast_divmod_create_divisor(A,loc=B,ip=ip)
		if H is _A:self.num_heads_fdd=cute.fast_divmod_create_divisor(self.num_o_heads,loc=B,ip=ip)
		self._loc=B
	def __extract_mlir_values__(self):
		A,self._values_pos=[],[]
		for C in[self.num_seqs,self.num_q_heads,self.num_v_heads,self.is_GQA,self.is_persistent,self.single_sequence_direct,self.total_tiles,self.num_seqs_fdd,self.num_heads_fdd]:B=extract_mlir_values(C);A+=B;self._values_pos.append(len(B))
		return A
	def __new_from_mlir_values__(self,values):
		A=values;B=[]
		for(D,C)in zip([self.num_seqs,self.num_q_heads,self.num_v_heads,self.is_GQA,self.is_persistent,self.single_sequence_direct,self.total_tiles,self.num_seqs_fdd,self.num_heads_fdd],self._values_pos,strict=_B):B.append(new_from_mlir_values(D,A[:C]));A=A[C:]
		return GDNTileSchedulerParams(*tuple(B),loc=self._loc)
	@dsl_user_op
	def get_grid_shape(self,max_active_clusters:Int32,*,loc=_A,ip=_A)->Tuple[Integer,Integer,Integer]:
		if cutlass.const_expr(self.is_persistent):return min(self.total_tiles,max_active_clusters),1,1
		else:return self.num_seqs,self.num_o_heads,1
class GDNTileScheduler:
	def __init__(self,params:GDNTileSchedulerParams,block_idx:Tuple[Integer,Integer,Integer],num_persistent_ctas:Int32,current_work_linear_idx:Int32,is_valid:bool=_B):self.params=params;self.block_idx=block_idx;self.num_persistent_ctas=num_persistent_ctas;self._current_work_linear_idx=current_work_linear_idx;self.is_valid=is_valid
	def __extract_mlir_values__(self)->list[ir.Value]:
		A,self._values_pos=[],[]
		for C in[self.params,self.block_idx,self.num_persistent_ctas,self._current_work_linear_idx,self.is_valid]:B=extract_mlir_values(C);A+=B;self._values_pos.append(len(B))
		return A
	def __new_from_mlir_values__(self,values:list[ir.Value])->_E:
		A=values;B=[]
		for(D,C)in zip([self.params,self.block_idx,self.num_persistent_ctas,self._current_work_linear_idx,self.is_valid],self._values_pos,strict=_B):B.append(new_from_mlir_values(D,A[:C]));A=A[C:]
		return GDNTileScheduler(*B)
	@staticmethod
	@dsl_user_op
	def create(params:GDNTileSchedulerParams,block_idx:Tuple[Integer,Integer,Integer],grid_dim:Tuple[Integer,Integer,Integer],*,loc=_A,ip=_A)->_E:A=block_idx;B=Int32(cute.size(grid_dim,loc=loc,ip=ip));C,D,D=A;return GDNTileScheduler(params,A,B,current_work_linear_idx=Int32(C),is_valid=_B)
	@staticmethod
	def get_grid_shape(params:GDNTileSchedulerParams,max_active_clusters:Int32,*,loc=_A,ip=_A)->Tuple[Integer,Integer,Integer]:return params.get_grid_shape(max_active_clusters,loc=loc,ip=ip)
	def _get_current_work_for_linear_idx(self,linear_idx:Int32,*,loc=_A,ip=_A)->WorkTileInfo:
		A=linear_idx;C=A<cute.size(self.params.total_tiles,loc=loc,ip=ip)
		if cutlass.const_expr(self.params.single_sequence_direct):B=Int32(0),Int32(A),Int32(1);return WorkTileInfo(B,C)
		D,E=divmod(A,self.params.num_heads_fdd);G,F=divmod(D,self.params.num_seqs_fdd);B=tuple(Int32(A)for A in(F,E,1));return WorkTileInfo(B,C)
	@dsl_user_op
	def get_current_work(self,*,loc=_A,ip=_A)->WorkTileInfo:
		if self.params.is_persistent:return self._get_current_work_for_linear_idx(self._current_work_linear_idx,loc=loc,ip=ip)
		A,B,C=self.block_idx;return WorkTileInfo((A,B,self.block_idx[2]),cutlass.Boolean(self.is_valid))
	@dsl_user_op
	def initial_work_tile_info(self,*,loc=_A,ip=_A)->WorkTileInfo:return self.get_current_work(loc=loc,ip=ip)
	@dsl_user_op
	def advance_to_next_work(self,*,advance_count:int=1,loc=_A,ip=_A):
		if self.params.is_persistent:self._current_work_linear_idx+=Int32(advance_count)*Int32(self.num_persistent_ctas)
		else:self.is_valid=_C
import math
from typing import Optional,Type,Tuple
import cuda.bindings.driver as cuda,cutlass,cutlass.cute as cute,cutlass.utils as utils
from cutlass.utils import TensorMapManager,TensorMapUpdateMode
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive,pipeline_init_wait
from cutlass.cute.nvgpu import cpasync,tcgen05,OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils,cutlass.cute.testing as testing
try:from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
	from cutlass.base_dsl import extract_mlir_values as _emv,new_from_mlir_values as _nfmv,extract_mlir_attributes as _ema,get_mlir_types as _gmt
	class TmaInfo:
		def __init__(self,atom,tma_tensor,smem_layout=_A):self._atom=atom;self._tma_tensor=tma_tensor
		@property
		def atom(self):return self._atom
		@property
		def tma_tensor(self):return self._tma_tensor
		def __extract_mlir_values__(self):return _emv(self._atom)+_emv(self._tma_tensor)
		def __extract_mlir_attributes__(self):return _ema(self._atom)+_ema(self._tma_tensor)
		def __new_from_mlir_values__(self,values):A=values;B=len(_gmt(self._atom));return TmaInfo(_nfmv(self._atom,A[:B]),_nfmv(self._tma_tensor,A[B:]))
		def __iter__(self):yield self._atom;yield self._tma_tensor
		def __getitem__(self,i):return(self._atom,self._tma_tensor)[i]
		def __len__(self):return 2
def _wrap_tma(ret):
	A=ret
	if isinstance(A,TmaInfo):return A
	return TmaInfo(A[0],A[1])
class GatedDeltaNetChunkedKernel:
	arch='sm_100';bytes_per_tensormap=128;num_tensormaps=4
	def __init__(self,io_dtype:Type[cutlass.Numeric],acc_dtype:Type[cutlass.Numeric],state_dtype:Type[cutlass.Numeric],mma_tiler_qk:Tuple[int,int,int],mma_tiler_qs:Tuple[int,int,int],mma_tiler_qkv:Tuple[int,int,int],mma_tiler_kv:Tuple[int,int,int],max_active_clusters:int,num_sm:int,is_GQA:bool,use_initial_state:bool,inverse_dtype:Type[cutlass.Numeric]=cutlass.Float16,store_final_state:bool=_B,enable_checkpoints:bool=_C,is_persistent:bool=_B,use_static_tma_descriptors:bool=_C,static_tma_full_chunks:bool=_C,use_static_scale:bool=_C,compact_cg1_events:bool=_C):
		A=mma_tiler_kv;self.io_dtype=io_dtype;self.acc_dtype=acc_dtype;self.state_dtype=state_dtype;self.mma_tiler_qk=mma_tiler_qk;self.mma_tiler_qs=mma_tiler_qs;self.mma_tiler_qkv=mma_tiler_qkv;self.mma_tiler_kv=A;self.split_value_rows=A[0]==64;self.max_active_clusters=max_active_clusters;self.num_sm=num_sm;self.is_GQA=is_GQA;self.inverse_dtype=inverse_dtype;self.use_initial_state=use_initial_state;self.store_final_state=store_final_state;self.enable_checkpoints=enable_checkpoints;self.is_persistent=is_persistent;self.use_static_tma_descriptors=use_static_tma_descriptors;self.static_tma_full_chunks=static_tma_full_chunks;self.use_static_scale=use_static_scale;self.compact_cg1_events=compact_cg1_events;self.use_scalar_tma=self.split_value_rows and self.use_static_tma_descriptors;self.use_ready_only_scalar=self.use_scalar_tma and self.compact_cg1_events;self.compute_group_0_warp_ids=[0,1,2,3];self.compute_group_1_warp_ids=[4,5,6,7];self.mma_warp_id=8;self.tma_qkv_warp_id=9;self.mma_cg1_warp_id=10;self.epilogue_warp_id=11;self.load_gate_beta_warp_id=self.epilogue_warp_id;self.num_regs_compute_group_0=224;self.num_regs_compute_group_1=256;self.num_regs_other=24
		if not self.use_initial_state:self.num_regs_compute_group_1=232;self.num_regs_other=48
		self.threads_per_cta=32*(len((self.mma_warp_id,self.tma_qkv_warp_id,self.mma_cg1_warp_id,self.epilogue_warp_id))+len(self.compute_group_0_warp_ids)+len(self.compute_group_1_warp_ids));self.use_2cta_instrs=_C;self.cluster_shape_mnk=1,1,1;self.cta_group=tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE;self.occupancy=1;self.threads_per_warp=32;self.tmem_alloc_cols=cute.arch.get_max_tmem_alloc_cols(self.arch);self.tmem_alloc_barrier=pipeline.NamedBarrier(barrier_id=1,num_threads=self.threads_per_warp*len((self.mma_warp_id,self.mma_cg1_warp_id,*self.compute_group_0_warp_ids,*self.compute_group_1_warp_ids)));self.inverse_barrier=pipeline.NamedBarrier(barrier_id=2,num_threads=self.threads_per_warp*len(self.compute_group_0_warp_ids));self.init_state_store_barrier=pipeline.NamedBarrier(barrier_id=4,num_threads=self.threads_per_warp*len(self.compute_group_1_warp_ids))
	def _setup_attributes(self):self.smem_q_stages=3;self.smem_k_stages=4 if self.split_value_rows else 3;self.smem_v_stages=2;self.smem_o_stages=1;self.smem_ainv_stages=3;self.smem_qk_stages=2;self.smem_gate_stages=5;self.smem_beta_stages=5;self.tmem_kv_acc_stages=1;self.tmem_q_state_acc_stages=1;self.tmem_state_inp_stages=1;self.tmem_shared_inp_stages=2;self.tmem_cg0_shared_acc_stages=2;self.tmem_cg1_shared_acc_stages=1;self.tmem_state_offset=0;self.tmem_q_state_offset=self.tmem_state_offset+self.tmem_kv_acc_stages*128;self.tmem_state_inp_offset=self.tmem_q_state_offset+self.tmem_q_state_acc_stages*64;self.tmem_cg0_shared_acc_offset=self.tmem_state_inp_offset+self.tmem_state_inp_stages*64;self.tmem_cg1_shared_acc_offset=self.tmem_cg0_shared_acc_offset+self.tmem_cg0_shared_acc_stages*64;self.tmem_shared_inp_offset=self.tmem_cg1_shared_acc_offset+self.tmem_cg1_shared_acc_stages*64;self.buffer_align_bytes=1024
	@staticmethod
	def can_implement(io_dtype,acc_dtype,inverse_dtype,mma_tiler_qk,mma_tiler_qs,mma_tiler_qkv,mma_tiler_kv):
		G=mma_tiler_qk;F=inverse_dtype;E=acc_dtype;D=mma_tiler_kv;C=mma_tiler_qkv;B=mma_tiler_qs;A=io_dtype
		if A not in[cutlass.Float16,cutlass.BFloat16]:raise testing.CantImplementError(f"io_dtype={A} not supported; only Float16 and BFloat16 are supported")
		if E!=cutlass.Float32:raise testing.CantImplementError(f"acc_dtype={E} not supported; only Float32 is supported")
		if F!=A:raise testing.CantImplementError(f"inverse_dtype={F} must match io_dtype={A}")
		if G!=(64,64,128):raise testing.CantImplementError(f"mma_tiler_qk={G} not supported; only (64, 64, 128) is supported")
		if B not in((128,64,128),(64,64,128)):raise testing.CantImplementError(f"mma_tiler_qs={B} not supported")
		if C not in((128,64,64),(64,64,64)):raise testing.CantImplementError(f"mma_tiler_qkv={C} not supported")
		if D not in((128,128,64),(64,128,64)):raise testing.CantImplementError(f"mma_tiler_kv={D} not supported")
		H=D[0]
		if B[0]!=H or C[0]!=H:raise testing.CantImplementError('all state/value MMA tiles must use one row extent')
	@cute.jit
	def __call__(self,q:cute.Tensor,k:cute.Tensor,v:cute.Tensor,gate:cute.Tensor,beta:cute.Tensor,o:cute.Tensor,cu_seqlens:cute.Tensor,s_in:Optional[cute.Tensor],s_out:Optional[cute.Tensor],s_indices:Optional[cute.Tensor],s_checkpoints:Optional[cute.Tensor],cu_checkpoints:Optional[cute.Tensor],checkpoint_every_n_tokens:cutlass.Int32,scale:cutlass.Float32,tensormap_workspace:cute.Tensor,stream:cuda.CUstream):
		V=cu_seqlens;H=s_checkpoints;E=beta;D=gate;C=s_out;B=s_in;self.b_t=64;G=q.shape[1];I=v.shape[1];h=V.shape[0]-1;i=D;j=E;self._setup_attributes()
		if cutlass.const_expr(self.split_value_rows):A=I//G;F=G;J=2;q=cute.make_tensor(q.iterator,cute.make_layout((q.shape[0],q.shape[2],(J,A,G)),stride=(q.stride[0],q.stride[2],(0,0,q.stride[1]))));k=cute.make_tensor(k.iterator,cute.make_layout((k.shape[0],k.shape[2],(J,A,G)),stride=(k.stride[0],k.stride[2],(0,0,k.stride[1]))));v=cute.make_tensor(v.iterator,cute.make_layout((self.mma_tiler_kv[0],v.shape[0],(J,A,G)),stride=(v.stride[2],v.stride[0],(self.mma_tiler_kv[0]*v.stride[2],v.stride[1],A*v.stride[1]))))
		elif cutlass.const_expr(self.is_GQA):A=G//I;F=I;q=cute.make_tensor(q.iterator,cute.make_layout((q.shape[0],q.shape[2],(A,I)),stride=(q.stride[0],q.stride[2],(q.stride[1],A*q.stride[1]))));k=cute.make_tensor(k.iterator,cute.make_layout((k.shape[0],k.shape[2],(A,I)),stride=(k.stride[0],k.stride[2],(0,k.stride[1]))));v=cute.make_tensor(v.iterator,cute.make_layout((v.shape[2],v.shape[0],(A,I)),stride=(v.stride[2],v.stride[0],(0,v.stride[1]))))
		else:A=I//G;F=G;q=cute.make_tensor(q.iterator,cute.make_layout((q.shape[0],q.shape[2],(A,G)),stride=(q.stride[0],q.stride[2],(0,q.stride[1]))));k=cute.make_tensor(k.iterator,cute.make_layout((k.shape[0],k.shape[2],(A,G)),stride=(k.stride[0],k.stride[2],(0,k.stride[1]))));v=cute.make_tensor(v.iterator,cute.make_layout((v.shape[2],v.shape[0],(A,G)),stride=(v.stride[2],v.stride[0],(v.stride[1],A*v.stride[1]))))
		if cutlass.const_expr(self.split_value_rows):D=cute.make_tensor(D.iterator,cute.make_layout((D.shape[0],(J,A,F)),stride=(D.stride[0],(0,D.stride[1],A*D.stride[1]))));E=cute.make_tensor(E.iterator,cute.make_layout((E.shape[0],(J,A,F)),stride=(E.stride[0],(0,E.stride[1],A*E.stride[1]))));o=cute.make_tensor(o.iterator,cute.make_layout((self.mma_tiler_kv[0],o.shape[0],(J,A,F)),stride=(o.stride[2],o.stride[0],(self.mma_tiler_kv[0]*o.stride[2],o.stride[1],A*o.stride[1]))))
		else:D=cute.make_tensor(D.iterator,cute.make_layout((D.shape[0],(A,F)),stride=(D.stride[0],(D.stride[1],A*D.stride[1]))));E=cute.make_tensor(E.iterator,cute.make_layout((E.shape[0],(A,F)),stride=(E.stride[0],(E.stride[1],A*E.stride[1]))));o=cute.make_tensor(o.iterator,cute.make_layout((o.shape[2],o.shape[0],(A,F)),stride=(o.stride[2],o.stride[0],(o.stride[1],A*o.stride[1]))))
		if cutlass.const_expr(B is not _A and self.split_value_rows):B=cute.make_tensor(B.iterator,cute.make_layout((self.mma_tiler_kv[0],B.shape[3],(J,A,F),B.shape[0]),stride=(B.stride[2],B.stride[3],(self.mma_tiler_kv[0]*B.stride[2],B.stride[1],A*B.stride[1]),B.stride[0])))
		elif cutlass.const_expr(B is not _A):B=cute.make_tensor(B.iterator,cute.make_layout((B.shape[2],B.shape[3],(A,F),B.shape[0]),stride=(B.stride[2],B.stride[3],(B.stride[1],A*B.stride[1]),B.stride[0])))
		if cutlass.const_expr(C is not _A and self.split_value_rows):C=cute.make_tensor(C.iterator,cute.make_layout((self.mma_tiler_kv[0],C.shape[3],(J,A,F),C.shape[0]),stride=(C.stride[2],C.stride[3],(self.mma_tiler_kv[0]*C.stride[2],C.stride[1],A*C.stride[1]),C.stride[0])))
		elif cutlass.const_expr(C is not _A):C=cute.make_tensor(C.iterator,cute.make_layout((C.shape[2],C.shape[3],(A,F),C.shape[0]),stride=(C.stride[2],C.stride[3],(C.stride[1],A*C.stride[1]),C.stride[0])))
		if cutlass.const_expr(self.enable_checkpoints):H=cute.make_tensor(H.iterator,cute.make_layout((H.shape[2],H.shape[3],(A,F),H.shape[0]),stride=(H.stride[2],H.stride[3],(H.stride[1],A*H.stride[1]),H.stride[0])))
		def L(mma_tiler,a_major,b_major,OperandSourceA):A=mma_tiler;return tcgen05.MmaF16BF16Op(self.io_dtype,self.acc_dtype,(A[0],A[1],16),self.cta_group,OperandSourceA,a_major,b_major)
		K=cute.make_tiled_mma(L(self.mma_tiler_qk,OperandMajorMode.K,OperandMajorMode.K,tcgen05.OperandSource.SMEM));l=cute.make_tiled_mma(L(self.mma_tiler_qs,OperandMajorMode.K,OperandMajorMode.K,tcgen05.OperandSource.TMEM));O=cute.make_tiled_mma(L(self.mma_tiler_qkv,OperandMajorMode.K,OperandMajorMode.K,tcgen05.OperandSource.TMEM));P=cute.make_tiled_mma(L(self.mma_tiler_qkv,OperandMajorMode.MN,OperandMajorMode.K,tcgen05.OperandSource.SMEM));W=cute.make_tiled_mma(L(self.mma_tiler_kv,OperandMajorMode.K,OperandMajorMode.MN,tcgen05.OperandSource.TMEM));self.cluster_layout_vmnk=cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk),(K.thr_id.shape,));M=cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group);m=cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp();Q=sm100_utils.make_smem_layout_a(K,self.mma_tiler_qk,self.io_dtype,self.smem_q_stages);R=sm100_utils.make_smem_layout_b(K,self.mma_tiler_qk,self.io_dtype,self.smem_k_stages);n=sm100_utils.make_smem_layout_b(W,self.mma_tiler_kv,self.io_dtype,self.smem_k_stages);S=sm100_utils.make_smem_layout_a(P,self.mma_tiler_qkv,self.io_dtype,self.smem_v_stages);X=sm100_utils.make_smem_layout_b(O,self.mma_tiler_qkv,self.io_dtype,self.smem_ainv_stages);Y=sm100_utils.make_smem_layout_b(O,self.mma_tiler_qkv,self.inverse_dtype,self.smem_ainv_stages);p=0 if self.inverse_dtype==self.io_dtype else cute.cosize(Y);Z=sm100_utils.make_smem_layout_b(O,self.mma_tiler_qkv,self.io_dtype,self.smem_qk_stages);T=sm100_utils.make_smem_layout_epi(self.io_dtype,cutlass.utils.LayoutEnum.from_tensor(o),self.mma_tiler_qkv[:2],self.smem_o_stages);U=cute.make_layout((self.b_t,1,self.smem_gate_stages));a=cute.make_layout((self.b_t,1,self.smem_beta_stages));N=cute.make_layout((self.b_t,4,2),stride=(4,1,self.b_t*4));r=cute.select(N,mode=[0,1]);b=cute.cosize(N)if self.use_scalar_tma else 0
		@cute.struct
		class s:load_k_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_k_stages*2];load_q_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_q_stages*2];load_v_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_v_stages*2];load_gate_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_gate_stages if self.use_ready_only_scalar else self.smem_gate_stages*2];load_beta_mbar_ptr:cute.struct.MemRange[cutlass.Int64,0 if self.use_ready_only_scalar else self.smem_beta_stages*2];raw_scalar_mbar_ptr:cute.struct.MemRange[cutlass.Int64,2 if self.use_scalar_tma else 0];q_state_acc_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.tmem_q_state_acc_stages*2];kv_acc_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.tmem_kv_acc_stages*2];cg0_shared_acc_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.tmem_cg0_shared_acc_stages*2];cg1_shared_acc_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.tmem_cg1_shared_acc_stages*2];ainv_ready_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_ainv_stages*2];qk_ready_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_qk_stages*2];state_inp_ready_mbar_ptr:cute.struct.MemRange[cutlass.Int64,1 if self.compact_cg1_events else self.tmem_state_inp_stages*2];vks_ready_mbar_ptr:cute.struct.MemRange[cutlass.Int64,0 if self.compact_cg1_events else 2];nv_ready_mbar_ptr:cute.struct.MemRange[cutlass.Int64,0 if self.compact_cg1_events else 2];decay_v_ready_mbar_ptr:cute.struct.MemRange[cutlass.Int64,0 if self.compact_cg1_events else 2];o_store_mbar_ptr:cute.struct.MemRange[cutlass.Int64,self.smem_o_stages*2];tmem_holding_buf:cutlass.Int32;sQ:cute.struct.Align[cute.struct.MemRange[self.io_dtype,cute.cosize(Q)],self.buffer_align_bytes];sK:cute.struct.Align[cute.struct.MemRange[self.io_dtype,cute.cosize(R)],self.buffer_align_bytes];sV:cute.struct.Align[cute.struct.MemRange[self.io_dtype,cute.cosize(S)],self.buffer_align_bytes];sAinv:cute.struct.Align[cute.struct.MemRange[self.io_dtype,cute.cosize(X)],self.buffer_align_bytes];sAinvCal:cute.struct.Align[cute.struct.MemRange[self.inverse_dtype,p],self.buffer_align_bytes];sQk:cute.struct.Align[cute.struct.MemRange[self.io_dtype,cute.cosize(Z)],self.buffer_align_bytes];sO:cute.struct.Align[cute.struct.MemRange[self.io_dtype,cute.cosize(T)],self.buffer_align_bytes];cumsumlog:cute.struct.MemRange[cutlass.Float32,cute.cosize(U)];cumprod:cute.struct.MemRange[cutlass.Float32,cute.cosize(U)];beta:cute.struct.MemRange[cutlass.Float32,cute.cosize(a)];raw_gate:cute.struct.Align[cute.struct.MemRange[cutlass.Float32,b],self.buffer_align_bytes];raw_beta:cute.struct.Align[cute.struct.MemRange[cutlass.Float32,b],self.buffer_align_bytes]
		self.shared_storage=s;c=cute.select(Q,mode=[0,1,2]);d=cute.select(R,mode=[0,1,2]);e=cute.select(S,mode=[0,1,2]);t=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(M,q,c,self.mma_tiler_qk,K,self.cluster_layout_vmnk.shape));u=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_B(M,k,d,self.mma_tiler_qk,K,self.cluster_layout_vmnk.shape));w=_wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(M,v,e,self.mma_tiler_qkv,P,self.cluster_layout_vmnk.shape));x=_wrap_tma(cpasync.make_tiled_tma_atom(M,i,N,(self.b_t,4)));y=_wrap_tma(cpasync.make_tiled_tma_atom(M,j,N,(self.b_t,4)));f=cute.select(T,mode=[0,1]);z=_wrap_tma(cpasync.make_tiled_tma_atom(m,o,f,self.mma_tiler_qkv[:2]));self.tma_q_bytes=cute.size_in_bytes(self.io_dtype,c);self.tma_k_bytes=cute.size_in_bytes(self.io_dtype,d);self.tma_v_bytes=cute.size_in_bytes(self.io_dtype,e);self.tma_o_bytes=cute.size_in_bytes(self.io_dtype,f);self.tma_scalar_bytes=2*cute.size_in_bytes(cutlass.Float32,r);g=GDNTileSchedulerParams(num_seqs=h,num_q_heads=G,num_v_heads=I*(2 if self.split_value_rows else 1),is_GQA=_C if self.split_value_rows else self.is_GQA,is_persistent=self.is_persistent,single_sequence_direct=self.use_static_tma_descriptors);A0=GDNTileScheduler.get_grid_shape(g,self.max_active_clusters);self.kernel(K,l,O,P,W,t,u,w,x,y,D,E,z,V,B,C,s_indices,H,cu_checkpoints,checkpoint_every_n_tokens,scale,Q,R,n,S,U,a,N,X,Y,Z,T,g,q,k,v,o,tensormap_workspace).launch(grid=A0,block=(self.threads_per_cta,1,1),cluster=self.cluster_shape_mnk,smem=self.shared_storage.size_in_bytes(),stream=stream,min_blocks_per_mp=1)
	@cute.kernel
	def kernel(self,tiled_mma_qk:cute.TiledMma,tiled_mma_qs:cute.TiledMma,tiled_mma_qkv:cute.TiledMma,tiled_mma_qkv_ss:cute.TiledMma,tiled_mma_kv:cute.TiledMma,tma_q:TmaInfo,tma_k:TmaInfo,tma_v:TmaInfo,tma_gate:TmaInfo,tma_beta:TmaInfo,mGate:cute.Tensor,mBeta:cute.Tensor,tma_o:TmaInfo,cu_seqlens:cute.Tensor,mS_init:Optional[cute.Tensor],mS_out:Optional[cute.Tensor],mS_indices:Optional[cute.Tensor],mS_checkpoints:Optional[cute.Tensor],cu_checkpoints:Optional[cute.Tensor],checkpoint_every_n_tokens:cutlass.Int32,scale:cutlass.Float32,q_smem_layout_staged:cute.ComposedLayout,k_smem_layout_staged:cute.ComposedLayout,k_trans_smem_layout_staged:cute.ComposedLayout,v_smem_layout_staged:cute.ComposedLayout,cumsumlog_smem_layout_staged:cute.Layout,beta_smem_layout_staged:cute.Layout,scalar_tma_smem_layout:cute.Layout,ainv_smem_layout_staged:cute.ComposedLayout,ainv_cal_smem_layout_staged:cute.ComposedLayout,qk_smem_layout_staged:cute.ComposedLayout,o_smem_layout_staged:cute.ComposedLayout,scheduler_params:GDNTileSchedulerParams,mQ,mK,mV,mO,tensormap_workspace:cute.Tensor):
		B5=ainv_cal_smem_layout_staged;B4=scalar_tma_smem_layout;B3=cumsumlog_smem_layout_staged;B2=k_trans_smem_layout_staged;B1=k_smem_layout_staged;B0=q_smem_layout_staged;A_=checkpoint_every_n_tokens;Az=mS_checkpoints;Am=o_smem_layout_staged;Al=qk_smem_layout_staged;Ak=ainv_smem_layout_staged;Aj=v_smem_layout_staged;Ai=scale;Ah=mS_indices;Ag=mBeta;Af=mGate;Ae=tiled_mma_qs;AW=mS_out;AV=mS_init;AU=tma_o;AT=tiled_mma_qkv;AB=tiled_mma_qkv_ss;AA=tiled_mma_qk;A0=tensormap_workspace;z=mO;y=mV;x=mK;w=scheduler_params;v=tma_v;u=tma_k;t=tma_q;c=tma_beta;b=tma_gate;a=tiled_mma_kv;N=mQ;M=cu_seqlens;Q,Y,Y=cute.arch.thread_idx();V=cute.arch.make_warp_uniform(cute.arch.warp_idx());j,k,l=cute.arch.block_idx();W=cute.arch.grid_dim()
		if cutlass.const_expr(self.use_initial_state):assert AV is not _A,'mS_init must be provided if use_initial_state is True'
		else:assert AV is _A,'mS_init must be None if use_initial_state is False'
		if cutlass.const_expr(self.store_final_state):assert AW is not _A,'mS_out must be provided if store_final_state is True'
		else:assert AW is _A,'mS_out must be None if store_final_state is False'
		if cutlass.const_expr(self.use_static_scale):Ai=cutlass.Float32(.08838834764831845)
		AX=l*W[1]*W[0]+k*W[0]+j;H=_A;A1=_A;A2=_A;A3=_A;AC=_A
		if cutlass.const_expr(not self.use_static_tma_descriptors):H=TensorMapManager(TensorMapUpdateMode.GMEM,self.bytes_per_tensormap);A0=self.initialize_workspace(A0,W);A1=H.get_tensormap_ptr(A0[AX,0,_A].iterator);A2=H.get_tensormap_ptr(A0[AX,1,_A].iterator);A3=H.get_tensormap_ptr(A0[AX,2,_A].iterator);AC=H.get_tensormap_ptr(A0[AX,3,_A].iterator)
		BL=cutlass.utils.SmemAllocator();A=BL.allocate(self.shared_storage);A4=A.sQ.get_tensor(B0.outer,swizzle=B0.inner);A5=A.sK.get_tensor(B1.outer,swizzle=B1.inner);B6=A.sK.get_tensor(B2.outer,swizzle=B2.inner);A6=A.sV.get_tensor(Aj.outer,swizzle=Aj.inner);AY=A.sAinv.get_tensor(Ak.outer,swizzle=Ak.inner)
		if cutlass.const_expr(self.inverse_dtype==self.io_dtype):B7=AY
		else:B7=A.sAinvCal.get_tensor(B5.outer,swizzle=B5.inner)
		An=A.sQk.get_tensor(Al.outer,swizzle=Al.inner);B8=A.sO.get_tensor(Am.outer,swizzle=Am.inner);d=A.cumsumlog.get_tensor(B3);m=A.cumprod.get_tensor(B3);e=A.beta.get_tensor(beta_smem_layout_staged);AD=_A;AE=_A
		if cutlass.const_expr(self.use_scalar_tma):AD=A.raw_gate.get_tensor(B4);AE=A.raw_beta.get_tensor(B4)
		if V==self.mma_warp_id:cpasync.prefetch_descriptor(t.atom);cpasync.prefetch_descriptor(u.atom);cpasync.prefetch_descriptor(v.atom);cpasync.prefetch_descriptor(AU.atom)
		if cutlass.const_expr(self.use_scalar_tma):
			if V==self.epilogue_warp_id:cpasync.prefetch_descriptor(b.atom);cpasync.prefetch_descriptor(c.atom)
		R=cutlass.utils.TmemAllocator(A.tmem_holding_buf.ptr,barrier_for_retrieve=self.tmem_alloc_barrier,allocator_warp_id=self.compute_group_1_warp_ids[0])
		def S(num_threads):return pipeline.CooperativeGroup(pipeline.Agent.Thread,num_threads)
		Ao=S(len([self.tma_qkv_warp_id]));Ap=S(self.threads_per_warp*len([self.load_gate_beta_warp_id]));X=S(len([self.mma_warp_id]));B9=S(len([self.mma_warp_id,self.mma_cg1_warp_id]));AF=S(self.threads_per_warp*len(self.compute_group_0_warp_ids));f=S(self.threads_per_warp*len(self.compute_group_1_warp_ids));BM=S(len(self.compute_group_1_warp_ids));BN=S(self.threads_per_warp*len(self.compute_group_0_warp_ids)+self.threads_per_warp*len(self.compute_group_1_warp_ids));BO=S(self.threads_per_warp*len([self.epilogue_warp_id]));n,A7=pipeline.PipelineTmaUmma.create(num_stages=self.smem_k_stages,producer_group=Ao,consumer_group=B9,tx_count=self.tma_k_bytes,barrier_storage=A.load_k_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();o,A8=pipeline.PipelineTmaUmma.create(num_stages=self.smem_q_stages,producer_group=Ao,consumer_group=B9,tx_count=self.tma_q_bytes,barrier_storage=A.load_q_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();p,BA=pipeline.PipelineTmaAsync.create(num_stages=self.smem_v_stages,producer_group=Ao,consumer_group=BM,tx_count=self.tma_v_bytes,barrier_storage=A.load_v_mbar_ptr.data_ptr(),defer_sync=_B).make_participants()
		if cutlass.const_expr(self.use_ready_only_scalar):BB=pipeline.MbarrierArray(barrier_storage=A.load_gate_mbar_ptr.data_ptr(),num_stages=self.smem_gate_stages,agent=(pipeline.PipelineOp.AsyncThread,S(1)));K=BB;AG=BB;J=_A;AZ=_A
		else:
			K,AG=pipeline.PipelineAsync.create(num_stages=self.smem_gate_stages,producer_group=Ap,consumer_group=BN,barrier_storage=A.load_gate_mbar_ptr.data_ptr(),defer_sync=_B).make_participants()
			if cutlass.const_expr(self.use_scalar_tma):J,AZ=pipeline.PipelineAsync.create(num_stages=self.smem_beta_stages,producer_group=Ap,consumer_group=AF,barrier_storage=A.load_beta_mbar_ptr.data_ptr(),defer_sync=_B).make_participants()
			else:J,AZ=pipeline.PipelineCpAsync.create(num_stages=self.smem_beta_stages,producer_group=Ap,consumer_group=AF,barrier_storage=A.load_beta_mbar_ptr.data_ptr(),defer_sync=_B).make_participants()
		Z,AH=pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_kv_acc_stages,producer_group=X,consumer_group=f,barrier_storage=A.kv_acc_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();AI,BC=pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_q_state_acc_stages,producer_group=X,consumer_group=f,barrier_storage=A.q_state_acc_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();Aq,BD=pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_cg0_shared_acc_stages,producer_group=X,consumer_group=AF,barrier_storage=A.cg0_shared_acc_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();AJ,BE=pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_cg1_shared_acc_stages,producer_group=X,consumer_group=f,barrier_storage=A.cg1_shared_acc_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();Ar,Aa=pipeline.PipelineAsyncUmma.create(num_stages=self.smem_ainv_stages,producer_group=AF,consumer_group=X,barrier_storage=A.ainv_ready_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();As,Ab=pipeline.PipelineAsyncUmma.create(num_stages=self.smem_qk_stages,producer_group=AF,consumer_group=X,barrier_storage=A.qk_ready_mbar_ptr.data_ptr(),defer_sync=_B).make_participants()
		if cutlass.const_expr(self.compact_cg1_events):g=pipeline.MbarrierArray(barrier_storage=A.state_inp_ready_mbar_ptr.data_ptr(),num_stages=1,agent=(pipeline.PipelineOp.AsyncThread,S(len(self.compute_group_1_warp_ids))));Ac=g;AK=g;At=g;AL=g;Au=g;AM=g;Av=g;AN=g
		else:Ac,AK=pipeline.PipelineAsyncUmma.create(num_stages=self.tmem_state_inp_stages,producer_group=f,consumer_group=X,barrier_storage=A.state_inp_ready_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();At,AL=pipeline.PipelineAsyncUmma.create(num_stages=1,producer_group=f,consumer_group=X,barrier_storage=A.vks_ready_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();Au,AM=pipeline.PipelineAsyncUmma.create(num_stages=1,producer_group=f,consumer_group=X,barrier_storage=A.nv_ready_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();Av,AN=pipeline.PipelineAsyncUmma.create(num_stages=1,producer_group=f,consumer_group=X,barrier_storage=A.decay_v_ready_mbar_ptr.data_ptr(),defer_sync=_B).make_participants()
		Aw,BF=pipeline.PipelineAsync.create(num_stages=self.smem_o_stages,producer_group=f,consumer_group=BO,barrier_storage=A.o_store_mbar_ptr.data_ptr(),defer_sync=_B).make_participants();pipeline_init_arrive(is_relaxed=_B);pipeline_init_wait()
		if V>=self.compute_group_0_warp_ids[0]and V<=self.compute_group_0_warp_ids[-1]:
			cute.arch.setmaxregister_increase(self.num_regs_compute_group_0);R.wait_for_alloc();T=R.retrieve_ptr(self.acc_dtype);D=GDNTileScheduler.create(w,(j,k,l),W);B=D.initial_work_tile_info()
			while B.is_valid_tile:
				C,G,Y=B.tile_idx
				if cutlass.const_expr(self.use_static_tma_descriptors):E=N.shape[0]
				else:F=M[C];E=M[C+1]-F
				q=2;U=cute.ceil_div(E,self.b_t*q);BP=self._transform_to_position_independent_layout(An,Al.inner);BQ=self._transform_to_position_independent_layout(AY,Ak.inner);BR=d,e,BQ,B7,BP
				for Ax in cutlass.range(U):AG,AZ,BD,Ar,As=self.compute_group_0_pair(Q,T,Ai,(AA,),BR,(AG,AZ,BD,Ar,As),(Ax==0,Ax<U-1,Ax))
				D.advance_to_next_work();B=D.get_current_work()
			Ar.tail();As.tail()
		if V>=self.compute_group_1_warp_ids[0]and V<=self.compute_group_1_warp_ids[-1]:
			cute.arch.setmaxregister_increase(self.num_regs_compute_group_1);R.allocate(self.tmem_alloc_cols);R.wait_for_alloc();T=R.retrieve_ptr(self.acc_dtype);D=GDNTileScheduler.create(w,(j,k,l),W);B=D.initial_work_tile_info()
			while B.is_valid_tile:
				C,G,Y=B.tile_idx
				if cutlass.const_expr(self.use_static_tma_descriptors):E=N.shape[0]
				else:F=M[C];E=M[C+1]-F
				O=cute.ceil_div(E,self.b_t)
				if O>0:
					Ad=0
					if cutlass.const_expr(self.enable_checkpoints):Ad=cu_checkpoints[C]
					if cutlass.const_expr(self.use_initial_state):Z=self._load_initial_state(Q,AV,Ah,G,C,T,a,Z)
					BS=self._transform_to_position_independent_layout(A6,Aj.inner);BT=self._transform_to_position_independent_layout(B8,Am.inner);U=cute.ceil_div(E,self.b_t*2);BU=U*2;AO=_B
					for BV in cutlass.range(BU):BA,AG,BE,AH,BC,Z,Ac,At,Au,Av,Aw,Ad=self.compute_group_1_chunk(Q,T,Ai,(a,Ae,AT),(BS,d,m,e,BT),(Az,Ad,A_),(BA,AG,BE,AH,BC,Z,Ac,At,Au,Av,Aw),(BV,U,G,E,AO));AO=_C
					if cutlass.const_expr(self.store_final_state or self.enable_checkpoints):AH=self._store_final_state(Q,AW,Ah,G,C,T,a,AH,E,Az,Ad,A_)
					else:BW=AH.wait_and_advance();BW.release()
				elif cutlass.const_expr(self.store_final_state and self.use_initial_state):self._store_empty_final_state(Q,AV,AW,Ah,G,C)
				D.advance_to_next_work();B=D.get_current_work()
			R.relinquish_alloc_permit();R.free(T);Aw.tail()
			if cutlass.const_expr(not self.compact_cg1_events):Ac.tail()
		elif V==self.mma_warp_id:
			cute.arch.setmaxregister_decrease(self.num_regs_other);R.wait_for_alloc();T=R.retrieve_ptr(self.acc_dtype);D=GDNTileScheduler.create(w,(j,k,l),W);B=D.initial_work_tile_info()
			while B.is_valid_tile:
				C,Y,Y=B.tile_idx
				if cutlass.const_expr(self.use_static_tma_descriptors):E=N.shape[0]
				else:F=M[C];E=M[C+1]-F
				U=cute.ceil_div(E,self.b_t*2)
				for Bf in cutlass.range(U):Aq,A7,A8=self.mma_cg0_pair(T,(AA,AT),(A4,A5),(Aq,A7,A8))
				D.advance_to_next_work();B=D.get_current_work()
			Aq.tail()
		elif V==self.mma_cg1_warp_id:
			cute.arch.setmaxregister_decrease(self.num_regs_other);R.wait_for_alloc();T=R.retrieve_ptr(self.acc_dtype);D=GDNTileScheduler.create(w,(j,k,l),W);B=D.initial_work_tile_info()
			while B.is_valid_tile:
				C,Y,Y=B.tile_idx
				if cutlass.const_expr(self.use_static_tma_descriptors):E=N.shape[0]
				else:F=M[C];E=M[C+1]-F
				BG=cute.ceil_div(E,self.b_t*2)*2;BH=BG>0
				if cutlass.const_expr(self.use_initial_state):BH=_B
				if BH:
					BI=0
					if cutlass.const_expr(not self.use_initial_state):AJ,AI,Z,A7,A8,Aa,Ab,AK,AL,AM,AN=self.mma_cg1_chunk(T,(Ae,AT,AB,a),(A4,A5,B6,A6,AY,An),(AJ,AI,Z,A7,A8,Aa,Ab,AK,AL,AM,AN),_B,cutlass.Int32(0));BI=1
					for L in cutlass.range(BI,BG):
						AO=_C
						if cutlass.const_expr(self.use_initial_state):AO=L==0
						AJ,AI,Z,A7,A8,Aa,Ab,AK,AL,AM,AN=self.mma_cg1_chunk(T,(Ae,AT,AB,a),(A4,A5,B6,A6,AY,An),(AJ,AI,Z,A7,A8,Aa,Ab,AK,AL,AM,AN),AO,L)
				D.advance_to_next_work();B=D.get_current_work()
			AJ.tail();AI.tail();Z.tail()
		elif V==self.tma_qkv_warp_id:
			cute.arch.setmaxregister_decrease(self.num_regs_other);D=GDNTileScheduler.create(w,(j,k,l),W);B=D.initial_work_tile_info()
			if cutlass.const_expr(not self.use_static_tma_descriptors):
				if B.is_valid_tile:H.init_tensormap_from_atom(t.atom,A1,self.tma_qkv_warp_id);H.init_tensormap_from_atom(u.atom,A2,self.tma_qkv_warp_id);H.init_tensormap_from_atom(v.atom,A3,self.tma_qkv_warp_id);H.fence_tensormap_initialization()
			while B.is_valid_tile:
				C,G,Y=B.tile_idx
				if cutlass.const_expr(self.use_static_tma_descriptors):F=cutlass.Int32(0);I=N.shape[0];E=I
				else:F=M[C];I=M[C+1];E=I-F
				q=2;U=cute.ceil_div(E,self.b_t*q);O=U*q
				if O>0:
					if cutlass.const_expr(not self.use_static_tma_descriptors):BX=cute.make_tensor(N.iterator,cute.make_layout((I,N.shape[1],N.shape[2]),stride=(N.stride[0],N.stride[1],N.stride[2])));BY=cute.make_tensor(x.iterator,cute.make_layout((I,x.shape[1],x.shape[2]),stride=(x.stride[0],x.stride[1],x.stride[2])));BZ=cute.make_tensor(y.iterator,cute.make_layout((y.shape[0],I,y.shape[2]),stride=(y.stride[0],y.stride[1],y.stride[2])));H.update_tensormap((BX,BY,BZ),(t.atom,u.atom,v.atom),(A1,A2,A3),self.tma_qkv_warp_id,(_A,_A,_A))
					r=cute.ceil_div(E,self.b_t)
					for L in cutlass.range(r-1):s=F+L*self.b_t;o,n,p=self.tma_qkv_warp((AA,AB,a),(t,u,v),(A4,A5,A6),(o,n,p),(s,L,C,G),(H,A1,A2,A3))
					L=r-1;s=F+L*self.b_t;o,n,p=self.tma_qkv_warp((AA,AB,a),(t,u,v),(A4,A5,A6),(o,n,p),(s,L,C,G),(H,A1,A2,A3))
					for L in cutlass.range(r,O):s=F+L*self.b_t;o,n,p=self.tma_qkv_warp((AA,AB,a),(t,u,v),(A4,A5,A6),(o,n,p),(s,L,C,G),(H,A1,A2,A3))
				D.advance_to_next_work();B=D.get_current_work()
			o.tail();n.tail();p.tail()
		if V==self.epilogue_warp_id:
			cute.arch.setmaxregister_decrease(self.num_regs_other);D=GDNTileScheduler.create(w,(j,k,l),W);B=D.initial_work_tile_info()
			if cutlass.const_expr(self.use_scalar_tma):
				BJ=self.b_t,4;Ba=cute.local_tile(b.tma_tensor,BJ,(_A,_A));Bb=cute.local_tile(c.tma_tensor,BJ,(_A,_A));BK=cute.make_layout(1);AP,AQ=cpasync.tma_partition(b.atom,0,BK,cute.group_modes(AD,0,2),cute.group_modes(Ba,0,2));AR,AS=cpasync.tma_partition(c.atom,0,BK,cute.group_modes(AE,0,2),cute.group_modes(Bb,0,2));Bc=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32);Ay=cute.make_tiled_copy_tv(Bc,cute.make_layout((self.threads_per_warp,),stride=(1,)),cute.make_layout((1,),stride=(1,)));A9=A.raw_scalar_mbar_ptr.data_ptr()
				with cute.arch.elect_one():
					for Bd in range(2):cute.arch.mbarrier_init(A9+Bd,1)
				cute.arch.mbarrier_init_fence();cute.arch.sync_warp()
			if cutlass.const_expr(not self.use_static_tma_descriptors):
				if B.is_valid_tile:H.init_tensormap_from_atom(AU.atom,AC,self.epilogue_warp_id);H.fence_tensormap_initialization()
			while B.is_valid_tile:
				C,G,Y=B.tile_idx
				if cutlass.const_expr(self.use_static_tma_descriptors):F=cutlass.Int32(0);I=N.shape[0];E=I
				else:F=M[C];I=M[C+1];E=I-F
				q=2;U=cute.ceil_div(E,self.b_t*q);O=U*q
				if O>0:
					r=cute.ceil_div(E,self.b_t)
					if cutlass.const_expr(not self.use_static_tma_descriptors):Be=cute.make_tensor(z.iterator,cute.make_layout((z.shape[0],I,z.shape[2]),stride=(z.stride[0],z.stride[1],z.stride[2])));H.update_tensormap((Be,),(AU.atom,),(AC,),self.epilogue_warp_id,(_A,));H.fence_tensormap_update(AC)
					if cutlass.const_expr(self.use_scalar_tma):self.issue_scalar_tma_warp((b.atom,c.atom),(AP,AQ,AR,AS),A9,(F,cutlass.Int32(0),G));self.issue_scalar_tma_warp((b.atom,c.atom),(AP,AQ,AR,AS),A9,(F+self.b_t,cutlass.Int32(1),G))
					for P in range(2):
						h=F+P*self.b_t;i=P>=r-1
						if cutlass.const_expr(self.use_scalar_tma):K,J=self.load_gate_beta_tma_warp(Q,(b.atom,c.atom,AP,AQ,AR,AS,Ay),(AD,AE,d,m,e),(A9,K,J),(h,G,i,I,P,O))
						else:K,J=self.load_gate_beta_warp(Q,(Af,Ag),(d,m,e),(K,J),(h,G,i,I))
					if O>2:
						for P in range(2,4):
							h=F+P*self.b_t;i=P>=r-1
							if cutlass.const_expr(self.use_scalar_tma):K,J=self.load_gate_beta_tma_warp(Q,(b.atom,c.atom,AP,AQ,AR,AS,Ay),(AD,AE,d,m,e),(A9,K,J),(h,G,i,I,P,O))
							else:K,J=self.load_gate_beta_warp(Q,(Af,Ag),(d,m,e),(K,J),(h,G,i,I))
					for L in cutlass.range(O):
						P=L+4
						if P<O:
							h=F+P*self.b_t;i=P>=r-1
							if cutlass.const_expr(self.use_scalar_tma):K,J=self.load_gate_beta_tma_warp(Q,(b.atom,c.atom,AP,AQ,AR,AS,Ay),(AD,AE,d,m,e),(A9,K,J),(h,G,i,I,P,O))
							else:K,J=self.load_gate_beta_warp(Q,(Af,Ag),(d,m,e),(K,J),(h,G,i,I))
						s=F+L*self.b_t;BF=self.epilogue_warp((B8,),(AU,),(BF,),(G,s),(H,AC))
				D.advance_to_next_work();B=D.get_current_work()
			if cutlass.const_expr(not self.use_ready_only_scalar):K.tail();J.tail()
	@cute.jit
	def issue_scalar_tma_warp(self,tma_atoms:tuple,tma_partitions:tuple,raw_scalar_mbar_ptr,work_args:tuple)->_A:
		B=raw_scalar_mbar_ptr;E,F=tma_atoms;G,H,I,J=tma_partitions;K,L,M=work_args;A=L%2;N=M//2;C=N//4;D=K//self.b_t
		with cute.arch.elect_one():cute.arch.mbarrier_arrive_and_expect_tx(B+A,self.tma_scalar_bytes)
		cute.copy(E,H[_A,D,C],G[_A,A],tma_bar_ptr=B+A);cute.copy(F,J[_A,D,C],I[_A,A],tma_bar_ptr=B+A)
	@cute.jit
	def load_gate_beta_tma_warp(self,tidx:cutlass.Int32,tma_args:tuple,smem_args:tuple,pipeline_args:tuple,work_args:tuple)->tuple:
		b,c,d,e,f,g,C=tma_args;h,i,j,k,l=smem_args;O,K,P=pipeline_args;Q,R,S,m,G,n=work_args
		if cutlass.const_expr(self.static_tma_full_chunks):S=_C
		T=tidx%self.threads_per_warp;o=R//2;U=o%4;L=G%2;p=G//2%2;cute.arch.mbarrier_wait(O+L,p);D=C.get_slice(T);H=D.partition_S(h[_A,U,L]);M=D.partition_S(i[_A,U,L]);q=D.partition_D(j);r=D.partition_D(k);A=cute.make_rmem_tensor_like(H,self.acc_dtype);I=cute.make_rmem_tensor_like(M,self.acc_dtype);V=cute.make_rmem_tensor_like(H,self.acc_dtype);s=cute.make_identity_tensor((self.b_t,));N=D.partition_S(s);J=cute.make_rmem_tensor(((N.shape[0][1],),N.shape[1]),cutlass.Boolean)
		if S:
			for E in range(cute.size(J)):J[E]=cute.elem_less(Q+N[E][0],m)
			A.fill(1.);I.fill(_D);cute.copy(C,H,A,pred=J);cute.copy(C,M,I,pred=J)
		else:cute.copy(C,H,A);cute.copy(C,M,I)
		cute.arch.sync_warp();cute.arch.fence_proxy('async.shared',space='cta');W=G+2
		if W<n:self.issue_scalar_tma_warp((b,c),(d,e,f,g),O,(Q+self.b_t*2,W,R))
		for E in range(cute.size(A)):A[E]=cute.math.log2(A[E]+1e-10,fastmath=_B)
		for X in[1,2,4,8,16]:
			for B in range(cute.size(A)):
				t=cute.arch.shuffle_sync_up(A[B],X,mask=4294967295,mask_and_clamp=0)
				if T>=X:A[B]=A[B]+t
		for B in range(1,cute.size(A)):u=cute.arch.shuffle_sync(A[B-1],self.threads_per_warp-1,mask=4294967295,mask_and_clamp=self.threads_per_warp-1);A[B]+=u
		for B in range(cute.size(A)):V[B]=cute.math.exp2(A[B],fastmath=_B)
		F=G%self.smem_gate_stages
		if cutlass.const_expr(not self.use_ready_only_scalar):Y=K.acquire_and_advance();F=Y.index
		cute.copy(C,A,q[_A,_A,0,F]);cute.copy(C,V,r[_A,_A,0,F])
		if cutlass.const_expr(not self.use_ready_only_scalar):Y.commit()
		v=D.partition_D(l);Z=F
		if cutlass.const_expr(not self.use_ready_only_scalar):a=P.acquire_and_advance();Z=a.index
		cute.copy(C,I,v[_A,_A,0,Z])
		if cutlass.const_expr(self.use_ready_only_scalar):
			cute.arch.sync_warp()
			with cute.arch.elect_one():K.arrive(F,_A)
		else:a.commit()
		return K,P
	@cute.jit
	def tma_qkv_warp(self,mma_args:tuple,tma_args:tuple,smem_args:tuple,pipeline_args:tuple,work_args:tuple,tensormap_args:tuple)->tuple[pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer]:
		Z,a,p=mma_args;B,C,D=tma_args;b,c,d=smem_args;L,M,N=pipeline_args;H,I,q,J=work_args;A,O,P,Q=tensormap_args;K=cute.make_layout(1);R=Z.get_slice(0);e=a.get_slice(0);S=cute.select(self.mma_tiler_qk,mode=[0,2]);f=cute.select(self.mma_tiler_qkv,mode=[0,2]);g=cute.domain_offset((H,cutlass.Int32(0)),C.tma_tensor[_A,_A,J]);h=cute.flat_divide(g,S);i=R.partition_B(h);T,U=cpasync.tma_partition(C.atom,0,K,cute.group_modes(c,0,3),cute.group_modes(i,0,3));E=M.acquire_and_advance()
		if cutlass.const_expr(not self.use_static_tma_descriptors):
			if I==0:A.fence_tensormap_update(P)
		if cutlass.const_expr(self.use_static_tma_descriptors):cute.copy(C.atom,U[_A,0,0],T[_A,E.index],tma_bar_ptr=E.barrier)
		else:cute.copy(C.atom,U[_A,0,0],T[_A,E.index],tma_bar_ptr=E.barrier,tma_desc_ptr=A.get_tensormap_ptr(P,cute.AddressSpace.generic))
		j=cute.domain_offset((H,cutlass.Int32(0)),B.tma_tensor[_A,_A,J]);k=cute.flat_divide(j,S);l=R.partition_A(k);V,W=cpasync.tma_partition(B.atom,0,K,cute.group_modes(b,0,3),cute.group_modes(l,0,3));F=L.acquire_and_advance()
		if cutlass.const_expr(not self.use_static_tma_descriptors):
			if I==0:A.fence_tensormap_update(O)
		if cutlass.const_expr(self.use_static_tma_descriptors):cute.copy(B.atom,W[_A,0,0],V[_A,F.index],tma_bar_ptr=F.barrier)
		else:cute.copy(B.atom,W[_A,0,0],V[_A,F.index],tma_bar_ptr=F.barrier,tma_desc_ptr=A.get_tensormap_ptr(O,cute.AddressSpace.generic))
		m=cute.domain_offset((cutlass.Int32(0),H),D.tma_tensor[_A,_A,J]);n=cute.flat_divide(m,f);o=e.partition_A(n);X,Y=cpasync.tma_partition(D.atom,0,K,cute.group_modes(d,0,3),cute.group_modes(o,0,3));G=N.acquire_and_advance()
		if cutlass.const_expr(not self.use_static_tma_descriptors):
			if I==0:A.fence_tensormap_update(Q)
		if cutlass.const_expr(self.use_static_tma_descriptors):cute.copy(D.atom,Y[_A,0,0],X[_A,G.index],tma_bar_ptr=G.barrier)
		else:cute.copy(D.atom,Y[_A,0,0],X[_A,G.index],tma_bar_ptr=G.barrier,tma_desc_ptr=A.get_tensormap_ptr(Q,cute.AddressSpace.generic))
		return L,M,N
	@cute.jit
	def load_gate_beta_warp(self,tidx:cutlass.Int32,gmem_args:tuple,smem_args:tuple,pipeline_args:tuple,work_args:tuple)->tuple[pipeline.PipelineProducer,pipeline.PipelineProducer]:
		T,c=gmem_args;d,e,f=smem_args;U,V=pipeline_args;J,K,E,g=work_args
		if cutlass.const_expr(self.static_tma_full_chunks):E=_C
		L=tidx%self.threads_per_warp;M=cute.domain_offset((J,),T[_A,K]);N=cute.domain_offset((J,),cute.make_identity_tensor(T[_A,K].shape));O=cute.domain_offset((J,),c[_A,K]);M=cute.flat_divide(M,(self.b_t,))[_A,0];N=cute.flat_divide(N,(self.b_t,))[_A,0];O=cute.flat_divide(O,(self.b_t,))[_A,0];W=cute.make_layout((self.threads_per_warp,),stride=(1,));X=cute.make_layout((1,),stride=(1,));h=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32);C=cute.make_tiled_copy_tv(h,W,X);i=cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS),cutlass.Float32,num_bits_per_copy=32);P=cute.make_tiled_copy_tv(i,W,X);F=C.get_slice(L);G=F.partition_S(M);j=F.partition_D(d);k=F.partition_D(e);Y=P.get_slice(L);Z=Y.partition_S(O);Q=Y.partition_D(f);l=cute.make_rmem_tensor_like(G,self.acc_dtype);A=C.retile(l);m=cute.make_rmem_tensor_like(G,self.acc_dtype);a=C.retile(m);R=F.partition_S(N);H=cute.make_rmem_tensor(((R.shape[0][1],),R.shape[1]),cutlass.Boolean)
		if E:
			for D in range(cute.size(H)):H[D]=cute.elem_less(R[D][0],g)
		if E:A.fill(1.);cute.copy(C,G,A,pred=H)
		else:cute.copy(C,G,A)
		for D in range(cute.size(A)):A[D]=cute.math.log2(A[D]+1e-10,fastmath=_B)
		for b in[1,2,4,8,16]:
			for B in range(cute.size(A)):
				n=cute.arch.shuffle_sync_up(A[B],b,mask=4294967295,mask_and_clamp=0)
				if L>=b:A[B]=A[B]+n
		for B in range(1,cute.size(A)):o=cute.arch.shuffle_sync(A[B-1],self.threads_per_warp-1,mask=4294967295,mask_and_clamp=self.threads_per_warp-1);A[B]+=o
		for B in range(cute.size(A)):a[B]=cute.math.exp2(A[B],fastmath=_B)
		S=U.acquire_and_advance();cute.copy(C,A,j[_A,_A,0,S.index]);cute.copy(C,a,k[_A,_A,0,S.index]);S.commit();I=V.acquire_and_advance()
		if E:Q[_A,_A,0,I.index].fill(_D);cute.copy(P,Z,Q[_A,_A,0,I.index],pred=H)
		else:cute.copy(P,Z,Q[_A,_A,0,I.index])
		I.commit();return U,V
	@cute.jit
	def mma_cg0_pair(self,tmem_ptr:cutlass.Int64,mma_args:tuple,smem_args:tuple,pipeline_args:tuple)->tuple[pipeline.PipelineProducer,pipeline.PipelineConsumer,pipeline.PipelineConsumer]:
		B,P=mma_args;U,Q=smem_args;D,I,J=pipeline_args;V=P.partition_shape_C((self.mma_tiler_qkv[0],self.mma_tiler_qkv[1]));W=P.make_fragment_C(cute.append(V,self.tmem_cg0_shared_acc_stages));C=cute.make_tensor(tmem_ptr+self.tmem_cg0_shared_acc_offset,W.layout);K=B.make_fragment_A(Q);E=B.make_fragment_B(Q);R=B.make_fragment_A(U);F=cute.size(K,mode=[2]);L=D.acquire_and_advance();G=I.wait_and_advance()
		for A in cutlass.range(F,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,C[_A,_A,_A,L.index],K[_A,_A,A,G.index],E[_A,_A,A,G.index],C[_A,_A,_A,L.index])
		L.commit();M=D.acquire_and_advance();H=I.wait_and_advance()
		for A in cutlass.range(F,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,C[_A,_A,_A,M.index],K[_A,_A,A,H.index],E[_A,_A,A,H.index],C[_A,_A,_A,M.index])
		M.commit();S=J.wait_and_advance();N=D.acquire_and_advance()
		for A in cutlass.range(F,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,C[_A,_A,_A,N.index],R[_A,_A,A,S.index],E[_A,_A,A,G.index],C[_A,_A,_A,N.index])
		N.commit();T=J.wait_and_advance();O=D.acquire_and_advance()
		for A in cutlass.range(F,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,C[_A,_A,_A,O.index],R[_A,_A,A,T.index],E[_A,_A,A,H.index],C[_A,_A,_A,O.index])
		O.commit();S.release();T.release();G.release();H.release();return D,I,J
	@cute.jit
	def mma_cg1_chunk(self,tmem_ptr:cutlass.Int64,mma_args:tuple,smem_args:tuple,pipeline_args:tuple,is_first_chunk:cutlass.Boolean,chunk_idx:cutlass.Int32)->tuple:
		a=is_first_chunk;K=chunk_idx;E=tmem_ptr;C,B,_,F=mma_args;l,m,n,A5,o,p=smem_args;L,M,N,b,c,d,e,O,P,Q,f=pipeline_args;q=B.partition_shape_C((self.mma_tiler_qkv[0],self.mma_tiler_qkv[1]));r=B.make_fragment_C(cute.append(q,self.tmem_cg1_shared_acc_stages));D=cute.make_tensor(E+self.tmem_cg1_shared_acc_offset,r.layout);s=B.partition_shape_A((self.mma_tiler_qkv[0],self.mma_tiler_qkv[2]));t=B.make_fragment_A(cute.append(s,self.tmem_shared_inp_stages));I=cute.make_tensor(cute.recast_ptr(E+self.tmem_shared_inp_offset,dtype=self.io_dtype),t.layout);u=C.partition_shape_C((self.mma_tiler_qs[0],self.mma_tiler_qs[1]));v=C.make_fragment_C(cute.append(u,self.tmem_q_state_acc_stages));J=cute.make_tensor(E+self.tmem_q_state_offset,v.layout);w=F.partition_shape_C((self.mma_tiler_kv[0],self.mma_tiler_kv[1]));x=F.make_fragment_C(cute.append(w,self.tmem_kv_acc_stages));g=cute.make_tensor(E+self.tmem_state_offset,x.layout);y=C.partition_shape_A((self.mma_tiler_qs[0],self.mma_tiler_qs[2]));z=C.make_fragment_A(cute.append(y,self.tmem_state_inp_stages));R=cute.make_tensor(cute.recast_ptr(E+self.tmem_state_inp_offset,dtype=self.io_dtype),z.layout);A0=C.make_fragment_B(m);A1=C.make_fragment_B(l);S=B.make_fragment_B(o);A2=B.make_fragment_B(p);h=F.make_fragment_B(n);i=cute.size(R,mode=[2]);T=cute.size(S,mode=[2]);A3=cute.size(h,mode=[2]);U=b.wait_and_advance();j=c.wait_and_advance();G=a==_C
		if cutlass.const_expr(self.use_initial_state):G=_B
		if G:
			V=L.acquire_and_advance()
			if cutlass.const_expr(self.compact_cg1_events):O.wait(0,K&1)
			else:A4=O.wait_and_advance()
			for A in cutlass.range(i,unroll_full=_B):C.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(C,D[_A,_A,_A,V.index],R[_A,_A,A,0],A0[_A,_A,A,U.index],D[_A,_A,_A,V.index])
			V.commit();W=M.acquire_and_advance()
			for A in cutlass.range(i,unroll_full=_B):C.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(C,J[_A,_A,_A,W.index],R[_A,_A,A,0],A1[_A,_A,A,j.index],J[_A,_A,_A,W.index])
			W.commit()
			if cutlass.const_expr(not self.compact_cg1_events):A4.release()
		j.release();H=L.acquire_and_advance()
		if cutlass.const_expr(self.compact_cg1_events):P.wait(0,K+1&1)
		else:P.wait_and_advance()
		X=d.wait_and_advance()
		if G:
			for A in cutlass.range(T,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,H.index],I[_A,_A,A,0],S[_A,_A,A,X.index],D[_A,_A,_A,H.index])
		else:
			for A in cutlass.range(T,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,H.index],I[_A,_A,A,0],S[_A,_A,A,X.index],D[_A,_A,_A,H.index])
		H.commit();X.release();Y=M.acquire_and_advance();k=e.wait_and_advance()
		if cutlass.const_expr(self.compact_cg1_events):Q.wait(0,K&1)
		else:Q.wait_and_advance()
		for A in cutlass.range(T,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,G or A!=0);cute.gemm(B,J[_A,_A,_A,Y.index],I[_A,_A,A,0],A2[_A,_A,A,k.index],J[_A,_A,_A,Y.index])
		k.release();Y.commit()
		if cutlass.const_expr(self.use_initial_state):
			if a:N.advance()
		Z=N.acquire_and_advance()
		if cutlass.const_expr(not self.compact_cg1_events):f.wait_and_advance()
		for A in cutlass.range(A3,unroll_full=_B):F.set(tcgen05.Field.ACCUMULATE,G or A!=0);cute.gemm(F,g[_A,_A,_A,Z.index],I[_A,_A,A,1],h[_A,_A,A,U.index],g[_A,_A,_A,Z.index])
		Z.commit();U.release();return L,M,N,b,c,d,e,O,P,Q,f
	@cute.jit
	def mma_issuer_warp(self,tmem_ptr:cutlass.Int64,scheduler_params:GDNTileSchedulerParams,block_coord:tuple,grid_dim:tuple,cu_seqlens:cute.Tensor,mma_args:tuple,smem_args:tuple,pipeline_args:tuple):
		V=smem_args;U=mma_args;T=cu_seqlens;S=tmem_ptr;A,B,C,D,E,F,G,H,I,J,K,L,M=pipeline_args;N=E.clone();O=F.clone();P=GDNTileScheduler.create(scheduler_params,block_coord,grid_dim);Q=P.initial_work_tile_info()
		while Q.is_valid_tile:
			W,a,a=Q.tile_idx;b=T[W];c=T[W+1]-b;R=cute.ceil_div(c,self.b_t*2);d=R*2;X=0
			if cutlass.const_expr(not self.use_initial_state):A,B,C,D,E,N,F,O,G,H,I,J,K,L,M=self.mma_warp_chunk(S,U,V,(A,B,C,D,E,N,F,O,G,H,I,J,K,L,M),(0,R,_B));X=1
			for Y in cutlass.range(X,d):
				Z=_C
				if cutlass.const_expr(self.use_initial_state):Z=Y==0
				A,B,C,D,E,N,F,O,G,H,I,J,K,L,M=self.mma_warp_chunk(S,U,V,(A,B,C,D,E,N,F,O,G,H,I,J,K,L,M),(Y,R,Z))
			P.advance_to_next_work();Q=P.get_current_work()
		A.tail();B.tail();C.tail();D.tail()
	@cute.jit
	def mma_warp_chunk(self,tmem_ptr:cutlass.Int64,mma_args:tuple,smem_args:tuple,pipeline_args:tuple,work_args:tuple)->tuple[pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer]:
		G=tmem_ptr;B,E,C,T,H=mma_args;r,U,AD,AE,AF,AG=smem_args;F,V,W,X,L,P,Y,Z,AH,s,t,u,v,w,x=pipeline_args;y,AI,a=work_args;z=C.partition_shape_C((self.mma_tiler_qkv[0],self.mma_tiler_qkv[1]));AJ=C.make_fragment_C(cute.append(z,self.tmem_cg0_shared_acc_stages));D=cute.make_tensor(G+self.tmem_cg0_shared_acc_offset,AJ.layout);AK=C.make_fragment_C(cute.append(z,self.tmem_cg1_shared_acc_stages));I=cute.make_tensor(G+self.tmem_cg1_shared_acc_offset,AK.layout);AL=C.partition_shape_A((self.mma_tiler_qkv[0],self.mma_tiler_qkv[2]));AM=C.make_fragment_A(cute.append(AL,self.tmem_shared_inp_stages));b=cute.make_tensor(cute.recast_ptr(G+self.tmem_shared_inp_offset,dtype=self.io_dtype),AM.layout);AN=E.partition_shape_C((self.mma_tiler_qs[0],self.mma_tiler_qs[1]));AO=E.make_fragment_C(cute.append(AN,self.tmem_q_state_acc_stages));Q=cute.make_tensor(G+self.tmem_q_state_offset,AO.layout);AP=H.partition_shape_C((self.mma_tiler_kv[0],self.mma_tiler_kv[1]));AQ=H.make_fragment_C(cute.append(AP,self.tmem_kv_acc_stages));A0=cute.make_tensor(G+self.tmem_state_offset,AQ.layout);AR=E.partition_shape_A((self.mma_tiler_qs[0],self.mma_tiler_qs[2]));AS=E.make_fragment_A(cute.append(AR,self.tmem_state_inp_stages));AT=cute.make_tensor(cute.recast_ptr(G+self.tmem_state_inp_offset,dtype=self.io_dtype),AS.layout);M=B.make_fragment_A(U);J=B.make_fragment_B(U);A1=B.make_fragment_A(r);c=AT;AU=E.make_fragment_B(r);AV=E.make_fragment_B(U);d=C.make_fragment_B(AF);AW=b;AX=C.make_fragment_B(AG);AY=b;A2=H.make_fragment_B(AD);AZ=T.make_fragment_A(AE);K=cute.size(M,mode=[2]);A3=cute.size(c,mode=[2]);e=cute.size(d,mode=[2]);Aa=cute.size(A2,mode=[2]);A4=y&1==0;Ab=y<AI*2-2
		if A4:
			A5=P.clone();R=A5.current_handle();A6=A5.clone();A6.advance();S=A6.current_handle()
			if a:
				f=F.acquire_and_advance();R=L.wait_and_advance()
				for A in cutlass.range(K,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,f.index],M[_A,_A,A,R.index],J[_A,_A,A,R.index],D[_A,_A,_A,f.index])
				f.commit();g=F.acquire_and_advance();S=L.wait_and_advance()
				for A in cutlass.range(K,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,g.index],M[_A,_A,A,S.index],J[_A,_A,A,S.index],D[_A,_A,_A,g.index])
				g.commit()
			h=F.acquire_and_advance();Ac=Y.wait_and_advance()
			for A in cutlass.range(K,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,h.index],A1[_A,_A,A,Ac.index],J[_A,_A,A,R.index],D[_A,_A,_A,h.index])
			h.commit();i=F.acquire_and_advance();Ad=Y.wait_and_advance()
			for A in cutlass.range(K,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,i.index],A1[_A,_A,A,Ad.index],J[_A,_A,A,S.index],D[_A,_A,_A,i.index])
			i.commit()
		A7=P.current_handle();A8=Z.current_handle();A9=A7.index;Ae=A8.index;N=a==_C
		if cutlass.const_expr(self.use_initial_state):N=_B
		if N:
			j=V.acquire_and_advance();k=u.wait_and_advance()
			for A in cutlass.range(A3,unroll_full=_B):E.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(E,I[_A,_A,_A,j.index],c[_A,_A,A,k.index],AV[_A,_A,A,A9],I[_A,_A,_A,j.index])
			j.commit();l=W.acquire_and_advance()
			for A in cutlass.range(A3,unroll_full=_B):E.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(E,Q[_A,_A,_A,l.index],c[_A,_A,A,k.index],AU[_A,_A,A,Ae],Q[_A,_A,_A,l.index])
			l.commit();k.release()
		A8.release();Z.advance();O=V.acquire_and_advance();v.wait_and_advance();m=s.wait_and_advance()
		if N:
			for A in cutlass.range(e,unroll_full=_B):C.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(C,I[_A,_A,_A,O.index],b[_A,_A,A,0],d[_A,_A,A,m.index],I[_A,_A,_A,O.index])
		else:
			for A in cutlass.range(e,unroll_full=_B):T.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(T,I[_A,_A,_A,O.index],AZ[_A,_A,A,0],d[_A,_A,A,m.index],I[_A,_A,_A,O.index])
		O.commit();m.release()
		if A4:
			if Ab:
				n=F.acquire_and_advance();AA=L.wait_and_advance()
				for A in cutlass.range(K,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,n.index],M[_A,_A,A,AA.index],J[_A,_A,A,AA.index],D[_A,_A,_A,n.index])
				n.commit();o=F.acquire_and_advance();AB=L.wait_and_advance()
				for A in cutlass.range(K,unroll_full=_B):B.set(tcgen05.Field.ACCUMULATE,A!=0);cute.gemm(B,D[_A,_A,_A,o.index],M[_A,_A,A,AB.index],J[_A,_A,A,AB.index],D[_A,_A,_A,o.index])
				o.commit()
		p=W.acquire_and_advance();AC=t.wait_and_advance();w.wait_and_advance()
		for A in cutlass.range(e,unroll_full=_B):
			if N:C.set(tcgen05.Field.ACCUMULATE,_B)
			else:C.set(tcgen05.Field.ACCUMULATE,A!=0)
			cute.gemm(C,Q[_A,_A,_A,p.index],AW[_A,_A,A,0],AX[_A,_A,A,AC.index],Q[_A,_A,_A,p.index])
		AC.release();p.commit()
		if cutlass.const_expr(self.use_initial_state):
			if a:X.advance()
		q=X.acquire_and_advance();x.wait_and_advance()
		for A in cutlass.range(Aa,unroll_full=_B):
			if N:H.set(tcgen05.Field.ACCUMULATE,_B)
			else:H.set(tcgen05.Field.ACCUMULATE,A!=0)
			cute.gemm(H,A0[_A,_A,_A,q.index],AY[_A,_A,A,1],A2[_A,_A,A,A9],A0[_A,_A,_A,q.index])
		q.commit();A7.release();P.advance();return F,V,W,X,L,P,Y,Z,AH,s,t,u,v,w,x
	@cute.jit
	def compute_group_0_pair(self,tidx:cutlass.Int32,tmem_ptr:cutlass.Int64,scale:cutlass.Float32,mma_args:tuple,smem_args:tuple,pipeline_args:tuple,work_args:tuple)->tuple[pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineProducer,pipeline.PipelineProducer]:
		p=scale;c=tidx;q,=mma_args;r,L,A9,Q,AA=smem_args;M,d,N,e,f=pipeline_args;_,_,R=work_args;AB=self.threads_per_warp*len(self.compute_group_0_warp_ids);S=c%AB;AC=q.partition_shape_C((self.mma_tiler_qk[0],self.mma_tiler_qk[1]));s=q.make_fragment_C(AC);AD=cute.make_tensor(s.iterator,cute.flat_product(s.layout,cute.make_layout((self.tmem_cg0_shared_acc_stages,),stride=(1,))));t=cute.make_tensor(tmem_ptr+self.tmem_cg0_shared_acc_offset,AD.layout);AE=utils.gemm.sm100.transform_partitioned_tensor_layout(t);AF=cute.make_identity_tensor((self.mma_tiler_qk[0],self.mma_tiler_qk[1]));AG=t[(_A,_A),0,0,0]
		if cutlass.const_expr(self.split_value_rows):u=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),self.acc_dtype)
		else:u=cute.make_copy_atom(tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)),self.acc_dtype)
		G=tcgen05.make_tmem_copy(u,AG);v=G.get_slice(S);H=v.partition_S(AE);C=v.partition_D(AF)
		if cutlass.const_expr(self.split_value_rows):w=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.io_dtype,num_bits_per_copy=32)
		else:w=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.io_dtype,num_bits_per_copy=128)
		AH=cute.make_tiled_copy_D(w,G)
		if cutlass.const_expr(self.inverse_dtype==self.io_dtype):g=utils.gemm.sm100.transform_partitioned_tensor_layout(Q);I=AH;h=I.get_slice(S).partition_D(g)
		else:AI=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.inverse_dtype,num_bits_per_copy=128);I=cute.make_tiled_copy_D(AI,G);g=utils.gemm.sm100.transform_partitioned_tensor_layout(Q);h=I.get_slice(S).partition_D(g)
		if cutlass.const_expr(self.split_value_rows):x=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.io_dtype,num_bits_per_copy=32)
		else:x=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.io_dtype,num_bits_per_copy=128)
		T=cute.make_tiled_copy_D(x,G);AJ=utils.gemm.sm100.transform_partitioned_tensor_layout(AA);y=T.get_slice(S).partition_D(AJ)
		if cutlass.const_expr(self.use_ready_only_scalar):U=R*2%self.smem_gate_stages;V=(R*2+1)%self.smem_gate_stages;AK=R*2//self.smem_gate_stages&1;AL=(R*2+1)//self.smem_gate_stages&1;M.wait(U,AK);M.wait(V,AL)
		else:z=M.wait_and_advance();A0=M.wait_and_advance();U=z.index;V=A0.index
		W=r[_A,0,U];J=cute.make_rmem_tensor_like(C,self.acc_dtype);X=r[_A,0,V];K=cute.make_rmem_tensor_like(C,self.acc_dtype)
		if cutlass.const_expr(self.split_value_rows):
			for B in cutlass.range_constexpr(cute.size(C)):E=C[B];O=E[0]>=E[1];J[B]=cute.math.exp2(W[E[0]]-W[E[1]],fastmath=_B)if O else _D;K[B]=cute.math.exp2(X[E[0]]-X[E[1]],fastmath=_B)if O else _D
		else:
			i=C[0][0];AM=W[i];AN=X[i]
			for B in cutlass.range_constexpr(cute.size(C)):E=C[B];O=i>=E[1];J[B]=cute.math.exp2(AM-W[E[1]],fastmath=_B)if O else _D;K[B]=cute.math.exp2(AN-X[E[1]],fastmath=_B)if O else _D
		if cutlass.const_expr(not self.use_ready_only_scalar):z.release();A0.release();j=d.wait_and_advance();k=d.wait_and_advance();Y=j.index;Z=k.index
		else:j=_A;k=_A;Y=U;Z=V
		if cutlass.const_expr(self.split_value_rows):
			A1=cute.make_rmem_tensor_like(C,self.acc_dtype);A2=cute.make_rmem_tensor_like(C,self.acc_dtype)
			for B in cutlass.range_constexpr(cute.size(C)):E=C[B];A1[B]=L[E[0],0,Y];A2[B]=L[E[0],0,Z]
		else:AO=L[C[0][0],0,Y];AP=L[C[0][0],0,Z]
		l=e.acquire_and_advance();F=cute.make_rmem_tensor_like(C,self.acc_dtype);m=cute.make_rmem_tensor_like(F,self.inverse_dtype);A3=I.retile(m);A4=N.wait_and_advance()
		for A in cutlass.range(F.shape[2]):cute.copy(G,H[_A,0,A,A4.index],F[_A,0,A])
		cute.arch.fence_view_async_tmem_load();A4.release()
		for A in cutlass.range(F.shape[2]):
			for B in cutlass.range(cute.size(J.shape[0]),vectorize=_B,unroll_full=_B):
				if cutlass.const_expr(self.split_value_rows):P=A1[B,0,A]
				else:P=AO
				F[B,0,A]=F[B,0,A]*J[B,0,A]*P
			m[_A,0,A].store(F[_A,0,A].load().to(self.inverse_dtype));cute.copy(I,A3[_A,0,A],h[_A,0,A,l.index])
		n=e.acquire_and_advance();A5=N.wait_and_advance()
		for A in cutlass.range(F.shape[2]):cute.copy(G,H[_A,0,A,A5.index],F[_A,0,A])
		cute.arch.fence_view_async_tmem_load();A5.release()
		for A in cutlass.range(F.shape[2]):
			for B in cutlass.range(cute.size(K.shape[0]),vectorize=_B,unroll_full=_B):
				if cutlass.const_expr(self.split_value_rows):P=A2[B,0,A]
				else:P=AP
				F[B,0,A]=F[B,0,A]*K[B,0,A]*P
			m[_A,0,A].store(F[_A,0,A].load().to(self.inverse_dtype));cute.copy(I,A3[_A,0,A],h[_A,0,A,n.index])
		self._partial_pair_inverse(c,Q,l,n);self._finish_pair_inverse(c,(L,A9,Q),(G,C),(l,n,j,k,Y,Z));A6=f.acquire_and_advance();a=N.wait_and_advance();D=cute.make_rmem_tensor_like(C,self.acc_dtype);o=cute.make_rmem_tensor_like(D,self.io_dtype);A7=T.retile(o)
		if cutlass.const_expr(self.split_value_rows):
			for A in cutlass.range(D.shape[2]):cute.copy(G,H[_A,0,A,a.index],D[_A,0,A])
			cute.arch.fence_view_async_tmem_load();a.release()
		for A in cutlass.range(D.shape[2]):
			if cutlass.const_expr(not self.split_value_rows):cute.copy(G,H[_A,0,A,a.index],D[_A,0,A])
			for B in cutlass.range(cute.size(J.shape[0]),vectorize=_B,unroll_full=_B):D[B,0,A]=D[B,0,A]*J[B,0,A]*p
			o[_A,0,A].store(D[_A,0,A].load().to(self.io_dtype));cute.copy(T,A7[_A,0,A],y[_A,0,A,A6.index])
		cute.arch.fence_view_async_shared()
		if cutlass.const_expr(not self.split_value_rows):cute.arch.fence_view_async_tmem_load();a.release()
		A6.commit();A8=f.acquire_and_advance();b=N.wait_and_advance()
		if cutlass.const_expr(self.split_value_rows):
			for A in cutlass.range(D.shape[2]):cute.copy(G,H[_A,0,A,b.index],D[_A,0,A])
			cute.arch.fence_view_async_tmem_load();b.release()
		for A in cutlass.range(D.shape[2]):
			if cutlass.const_expr(not self.split_value_rows):cute.copy(G,H[_A,0,A,b.index],D[_A,0,A])
			for B in cutlass.range(cute.size(K.shape[0]),vectorize=_B,unroll_full=_B):D[B,0,A]=D[B,0,A]*K[B,0,A]*p
			o[_A,0,A].store(D[_A,0,A].load().to(self.io_dtype));cute.copy(T,A7[_A,0,A],y[_A,0,A,A8.index])
		cute.arch.fence_view_async_shared()
		if cutlass.const_expr(not self.split_value_rows):cute.arch.fence_view_async_tmem_load();b.release()
		A8.commit();return M,d,N,e,f
	@cute.jit
	def _partial_pair_inverse(self,tidx:cutlass.Int32,sAinvCal:cute.Tensor,ainv0_handle,ainv1_handle):
		F=ainv1_handle;E=ainv0_handle;I=self.threads_per_warp*len(self.compute_group_0_warp_ids);B=tidx%I;A=B//32;C=B%32;J=A//2;K=A%2;D=utils.gemm.sm100.transform_partitioned_tensor_layout(sAinvCal);L=D[_A,_A,E.index];M=D[_A,_A,F.index];G=E.index
		if J==1:G=F.index
		N=D[_A,_A,G];O=cute.flat_divide(N,(8,8));self.inverse_barrier.arrive_and_wait();H=(K*self.threads_per_warp+C)//8;self._invert_diagonal_NxN(O[_A,_A,H,H],B,8);self.inverse_barrier.arrive_and_wait();P=cute.flat_divide(L,(16,16));Q=cute.flat_divide(M,(16,16));self._blockwise_diagonal_8x8_to_16x16(P[_A,_A,A,A],C);self._blockwise_diagonal_8x8_to_16x16(Q[_A,_A,A,A],C);self.inverse_barrier.arrive_and_wait()
	@cute.jit
	def _finish_pair_inverse(self,tidx:cutlass.Int32,smem_args:tuple,copy_args:tuple,handle_args:tuple):
		R,e,f=smem_args;S,B=copy_args;C,D,g,h,i,j=handle_args;k=self.threads_per_warp*len(self.compute_group_0_warp_ids);E=tidx%k;T=E//32;U=E%32;V=utils.gemm.sm100.transform_partitioned_tensor_layout(e);M=utils.gemm.sm100.transform_partitioned_tensor_layout(f);l=T//2;N=T%2;W=C.index
		if l==1:W=D.index
		X=M[_A,_A,W];m=cute.flat_divide(X,(32,32));self._blockwise_diagonal_16x16_to_32x32(m[_A,_A,N,N],U);self.inverse_barrier.arrive_and_wait();n=cute.flat_divide(X,(64,64));self._blockwise_diagonal_32x32_to_64x64(n[_A,_A,0,0],N,U);self.inverse_barrier.arrive_and_wait();o=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.io_dtype,num_bits_per_copy=32 if cutlass.const_expr(self.split_value_rows)else 128);F=cute.make_tiled_copy_D(o,S);O=F.get_slice(E)
		if cutlass.const_expr(self.inverse_dtype==self.io_dtype):G=F;P=O
		else:p=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),self.inverse_dtype,num_bits_per_copy=128);G=cute.make_tiled_copy_D(p,S);P=G.get_slice(E)
		H=cute.make_rmem_tensor_like(B,self.acc_dtype);Y=O.partition_S(V)[_A,_A,_A,C.index];Z=P.partition_S(M)[_A,_A,_A,C.index];I=cute.make_rmem_tensor_like(Y,self.io_dtype);J=cute.make_rmem_tensor_like(I,self.acc_dtype);a=cute.make_rmem_tensor_like(Z,self.inverse_dtype)
		for A in cutlass.range(cute.size(B)):Q=B[A];H[A]=R[Q[1],0,i]
		cute.copy(G,Z,a);J.store(a.load().to(self.acc_dtype))
		for A in cutlass.range(cute.size(I),vectorize=_B,unroll_full=_B):J[A]=J[A]*H[A]
		I.store(J.load().to(self.io_dtype));cute.copy(F,I,Y);cute.arch.fence_view_async_shared();C.commit()
		if cutlass.const_expr(not self.use_ready_only_scalar):g.release()
		b=O.partition_S(V)[_A,_A,_A,D.index];c=P.partition_S(M)[_A,_A,_A,D.index];K=cute.make_rmem_tensor_like(b,self.io_dtype);L=cute.make_rmem_tensor_like(K,self.acc_dtype);d=cute.make_rmem_tensor_like(c,self.inverse_dtype)
		for A in cutlass.range(cute.size(B)):Q=B[A];H[A]=R[Q[1],0,j]
		cute.copy(G,c,d);L.store(d.load().to(self.acc_dtype))
		for A in cutlass.range(cute.size(K),vectorize=_B,unroll_full=_B):L[A]=L[A]*H[A]
		K.store(L.load().to(self.io_dtype));cute.copy(F,K,b);cute.arch.fence_view_async_shared();D.commit()
		if cutlass.const_expr(not self.use_ready_only_scalar):h.release()
	def _make_acc_tensor_into_a_view(self,acc:cute.Tensor)->cute.Tensor:A=cute.logical_divide(acc.layout,(_A,_A,2));B=cute.make_layout(((A.shape[0],A.shape[2][0]),A.shape[1],A.shape[2][1]),stride=((A.stride[0],A.stride[2][0]),A.stride[1],A.stride[2][1]));return cute.make_tensor(acc.iterator,B)
	def _convert_f32_fragment_to_tf32_operand(self,fragment:cute.Tensor)->cute.Tensor:
		A=fragment;D=A.load();B=cute.make_rmem_tensor_like(A,cutlass.Int32)
		for C in range(cute.size(A)):B[C]=cute.arch.cvt_f32_tf32(D[C])
		return B
	def _assume_tensor_iterator_alignment(self,tensor:cute.Tensor,alignment:int)->cute.Tensor:A=tensor;return cute.make_tensor(A.iterator.align(alignment),A.layout)
	@cute.jit
	def _invert_diagonal_NxN(self,mat_NxN,tidx,N:int=8):
		F=mat_NxN;D=tidx%N;E=cute.make_rmem_tensor((N,),F.element_type);cute.autovec_copy(F[D,_A],E);A=cute.make_rmem_tensor_like(E,self.acc_dtype);A.store(E.load().to(cutlass.Float32))
		for B in cutlass.range_constexpr(N):A[B]=1. if D==B else A[B]
		for C in cutlass.range_constexpr(N-1):
			G=-A[C]
			for B in cutlass.range_constexpr(C):H=cute.arch.shuffle_sync(A[B],C,mask=4294967295,mask_and_clamp=6175);A[B]=A[B]+G*H if D>C else A[B]
			A[C]=G if D>C else A[C]
		E.store(A.load().to(F.element_type));cute.autovec_copy(E,F[D,_A])
	@cute.jit
	def _blockwise_diagonal_8x8_to_16x16(self,mat_16x16,lane_id):
		G=lane_id;B=mat_16x16
		if cutlass.const_expr(B.element_type==cutlass.Float32):d=cute.nvgpu.warp.MmaTF32Op((16,8,8))
		else:d=cute.nvgpu.warp.MmaF16BF16Op(B.element_type,self.acc_dtype,(16,8,8))
		A=cute.make_tiled_mma(d,cute.make_layout((1,1,1)));X=A.get_slice(G)
		if cutlass.const_expr(B.element_type==cutlass.Float32):H=cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1,transpose=_C),cutlass.Float32),A);Y=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32),A);Q=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32),A);I=cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=64),A)
		else:H=cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1,transpose=_C),B.element_type),A);Y=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1,transpose=_B),B.element_type),A);Q=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1,transpose=_B),B.element_type),A);I=cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=1,transpose=_C),B.element_type),A)
		R=H.get_slice(G);e=Y.get_slice(G);f=Q.get_slice(G);Z=I.get_slice(G);S=cute.flat_divide(B,(8,8));g=S[_A,_A,1,1];J=S[_A,_A,1,0];J=cute.make_tensor(J.iterator,cute.select(J.layout,mode=[1,0]));K=S[_A,_A,0,0];K=cute.make_tensor(K.iterator,cute.select(K.layout,mode=[1,0]));h=S[_A,_A,1,0];T=cute.make_tensor(g.iterator,cute.logical_product(g.layout,(cute.make_layout((2,),stride=(0,)),)))
		if cutlass.const_expr(B.element_type==cutlass.Float32):T=self._assume_tensor_iterator_alignment(T,16)
		U=cute.make_tensor(h.iterator,cute.logical_product(h.layout,(cute.make_layout((2,),stride=(0,)),)))
		if cutlass.const_expr(B.element_type==cutlass.Float32):U=self._assume_tensor_iterator_alignment(U,16)
		L=A.make_fragment_A(X.partition_A(T));M=A.make_fragment_B(X.partition_B(J));N=A.make_fragment_B(X.partition_B(K))
		if cutlass.const_expr(B.element_type==cutlass.Float32):V=cute.make_rmem_tensor_like(L,cutlass.Float32);a=cute.make_rmem_tensor_like(M,cutlass.Float32);b=cute.make_rmem_tensor_like(N,cutlass.Float32)
		else:V=L;a=M;b=N
		C=A.make_fragment_C(A.partition_shape_C((16,8)));D=A.make_fragment_C(A.partition_shape_C((16,8)));E=R.partition_S(T);O=R.retile(V)
		if cutlass.const_expr(B.element_type==cutlass.Float32):E=self._assume_tensor_iterator_alignment(E,16)
		if cutlass.const_expr(B.element_type!=cutlass.Float32):E=cute.logical_divide(E,(E.shape[0],_A,_A));O=cute.logical_divide(O,(O.shape[0],_A,_A))
		q=e.partition_S(J);r=e.retile(a);i=f.partition_S(K);j=f.retile(b);P=Z.partition_D(U);k=P;P=cute.logical_divide(P,(P.shape[0],_A,_A));F=Z.retile(D);s=F;F=cute.logical_divide(F,(F.shape[0],_A,_A))
		if cutlass.const_expr(B.element_type==cutlass.Float32):cute.copy(H,E,O)
		else:cute.copy(H,E[(_A,0),_A,_A],O[(_A,0),_A,_A])
		cute.copy(Y,q,r)
		if cutlass.const_expr(B.element_type==cutlass.Float32):L=self._convert_f32_fragment_to_tf32_operand(V);M=self._convert_f32_fragment_to_tf32_operand(a);C.fill(_D);cute.gemm(A,C,L,M,C);l=Z.retile(C);m=cute.make_rmem_tensor_like(l,cutlass.Float32);m.store(cutlass.Float32(_D)-l.load());cute.copy(I,m,k);cute.arch.sync_warp();c=R.partition_S(U);n=cute.make_rmem_tensor_like(V,cutlass.Float32);t=R.retile(n);c=self._assume_tensor_iterator_alignment(c,16);cute.copy(H,c,t);W=self._convert_f32_fragment_to_tf32_operand(n);cute.copy(Q,i,j);N=self._convert_f32_fragment_to_tf32_operand(b);D.fill(_D);cute.gemm(A,D,W,N,D);cute.copy(I,s,k)
		else:C.fill(_D);cute.gemm(A,C,L,M,C);C.store(cutlass.Float32(_D)-C.load());o=self._make_acc_tensor_into_a_view(C);W=cute.make_rmem_tensor_like(o,B.element_type);W.store(o.load().to(B.element_type));cute.copy(Q,i,j);D.fill(_D);cute.gemm(A,D,W,N,D);p=cute.make_rmem_tensor_like(F[(_A,0),_A,_A],B.element_type);p.store(F[(_A,0),_A,_A].load().to(B.element_type));cute.copy(I,p,P[(_A,0),_A,_A])
	@cute.jit
	def _blockwise_diagonal_16x16_to_32x32(self,mat_32x32,lane_id):
		D=lane_id;B=mat_32x32
		if cutlass.const_expr(B.element_type==cutlass.Float32):W=cute.nvgpu.warp.MmaTF32Op((16,8,8))
		else:W=cute.nvgpu.warp.MmaF16BF16Op(B.element_type,self.acc_dtype,(16,8,16))
		A=cute.make_tiled_mma(W,cute.make_layout((1,1,1)),permutation_mnk=(16,16,16));P=A.get_slice(D)
		if cutlass.const_expr(B.element_type==cutlass.Float32):G=cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_C),cutlass.Float32),A);Q=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32),A);R=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32),A);H=cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=64),A)
		else:G=cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_C),B.element_type),A);Q=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_B),B.element_type),A);R=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_B),B.element_type),A);H=cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4,transpose=_C),B.element_type),A)
		I=G.get_slice(D);X=Q.get_slice(D);Y=R.get_slice(D);S=H.get_slice(D);J=cute.flat_divide(B,(16,16));Z=J[_A,_A,1,1];E=J[_A,_A,1,0];E=cute.make_tensor(E.iterator,cute.select(E.layout,mode=[1,0]));F=J[_A,_A,0,0];F=cute.make_tensor(F.iterator,cute.select(F.layout,mode=[1,0]));a=J[_A,_A,1,0];K=A.make_fragment_A(P.partition_A(Z));L=A.make_fragment_B(P.partition_B(E));M=A.make_fragment_B(P.partition_B(F))
		if cutlass.const_expr(B.element_type==cutlass.Float32):N=cute.make_rmem_tensor_like(K,cutlass.Float32);T=cute.make_rmem_tensor_like(L,cutlass.Float32);U=cute.make_rmem_tensor_like(M,cutlass.Float32)
		else:N=K;T=L;U=M
		C=A.make_fragment_C(A.partition_shape_C((16,16)));O=A.make_fragment_C(A.partition_shape_C((16,16)));i=I.partition_S(Z);j=I.retile(N);k=X.partition_S(E);l=X.retile(T);m=Y.partition_S(F);n=Y.retile(U);b=S.partition_D(a);c=S.retile(O);cute.copy(G,i,j);cute.copy(Q,k,l)
		if cutlass.const_expr(B.element_type==cutlass.Float32):K=self._convert_f32_fragment_to_tf32_operand(N);L=self._convert_f32_fragment_to_tf32_operand(T)
		C.fill(_D);cute.gemm(A,C,K,L,C)
		if cutlass.const_expr(B.element_type==cutlass.Float32):d=S.retile(C);e=cute.make_rmem_tensor_like(d,cutlass.Float32);e.store(cutlass.Float32(_D)-d.load());cute.copy(H,e,b);cute.arch.sync_warp();o=I.partition_S(a);f=cute.make_rmem_tensor_like(N,cutlass.Float32);p=I.retile(f);cute.copy(G,o,p);V=self._convert_f32_fragment_to_tf32_operand(f)
		else:C.store(-C.load());g=self._make_acc_tensor_into_a_view(C);V=cute.make_rmem_tensor_like(g,B.element_type);V.store(g.load().to(B.element_type))
		cute.copy(R,m,n)
		if cutlass.const_expr(B.element_type==cutlass.Float32):M=self._convert_f32_fragment_to_tf32_operand(U)
		O.fill(_D);cute.gemm(A,O,V,M,O);h=cute.make_rmem_tensor_like(c,B.element_type);h.store(c.load().to(B.element_type));cute.copy(H,h,b)
	@cute.jit
	def _blockwise_diagonal_32x32_to_64x64(self,mat_64x64,local_warp_id,lane_id):
		V=local_warp_id;D=lane_id;B=mat_64x64
		if cutlass.const_expr(B.element_type==cutlass.Float32):W=cute.nvgpu.warp.MmaTF32Op((16,8,8))
		else:W=cute.nvgpu.warp.MmaF16BF16Op(B.element_type,self.acc_dtype,(16,8,16))
		A=cute.make_tiled_mma(W,cute.make_layout((1,1,1)),permutation_mnk=(16,32,32));M=A.get_slice(D)
		if cutlass.const_expr(B.element_type==cutlass.Float32):E=cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_C),cutlass.Float32),A);N=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32),A);O=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=32),A);F=cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Float32,num_bits_per_copy=64),A)
		else:E=cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_C),B.element_type),A);N=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_B),B.element_type),A);O=cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_B),B.element_type),A);F=cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4,transpose=_C),B.element_type),A)
		G=E.get_slice(D);X=N.get_slice(D);Y=O.get_slice(D);P=F.get_slice(D);Q=cute.flat_divide(B,(32,32));l=Q[_A,_A,1,1];R=Q[_A,_A,1,0];Z=Q[_A,_A,0,0];a=cute.flat_divide(l,(16,32))[_A,_A,V,0];b=cute.make_tensor(R.iterator,cute.select(R.layout,mode=[1,0]));c=cute.make_tensor(Z.iterator,cute.select(Z.layout,mode=[1,0]));d=cute.flat_divide(R,(16,32))[_A,_A,V,0];H=A.make_fragment_A(M.partition_A(a));I=A.make_fragment_B(M.partition_B(b));J=A.make_fragment_B(M.partition_B(c))
		if cutlass.const_expr(B.element_type==cutlass.Float32):K=cute.make_rmem_tensor_like(H,cutlass.Float32);S=cute.make_rmem_tensor_like(I,cutlass.Float32);T=cute.make_rmem_tensor_like(J,cutlass.Float32)
		else:K=H;S=I;T=J
		C=A.make_fragment_C(A.partition_shape_C((16,32)));L=A.make_fragment_C(A.partition_shape_C((16,32)));m=G.partition_S(a);n=G.retile(K);o=X.partition_S(b);p=X.retile(S);q=Y.partition_S(c);r=Y.retile(T);e=P.partition_D(d);f=P.retile(L);cute.copy(E,m,n);cute.copy(N,o,p)
		if cutlass.const_expr(B.element_type==cutlass.Float32):H=self._convert_f32_fragment_to_tf32_operand(K);I=self._convert_f32_fragment_to_tf32_operand(S)
		C.fill(_D);cute.gemm(A,C,H,I,C)
		if cutlass.const_expr(B.element_type==cutlass.Float32):g=P.retile(C);h=cute.make_rmem_tensor_like(g,cutlass.Float32);h.store(cutlass.Float32(_D)-g.load());cute.copy(F,h,e);self.inverse_barrier.arrive_and_wait();s=G.partition_S(d);i=cute.make_rmem_tensor_like(K,cutlass.Float32);t=G.retile(i);cute.copy(E,s,t);U=self._convert_f32_fragment_to_tf32_operand(i)
		else:C.store(-C.load());j=self._make_acc_tensor_into_a_view(C);U=cute.make_rmem_tensor_like(j,B.element_type);U.store(j.load().to(B.element_type))
		cute.copy(O,q,r)
		if cutlass.const_expr(B.element_type==cutlass.Float32):J=self._convert_f32_fragment_to_tf32_operand(T)
		L.fill(_D);cute.gemm(A,L,U,J,L);k=cute.make_rmem_tensor_like(f,B.element_type);k.store(f.load().to(B.element_type));self.inverse_barrier.arrive_and_wait();cute.copy(F,k,e)
	@cute.jit
	def _load_initial_state(self,tidx,mS_init,mS_indices,head_idx,batch_idx,tmem_ptr,tiled_mma_kv,kv_acc_producer)->pipeline.PipelineProducer:
		H=kv_acc_producer;G=tiled_mma_kv;F=batch_idx;E=mS_indices;P=self.threads_per_warp*len(self.compute_group_1_warp_ids);I=tidx%P;Q=G.partition_shape_C((self.mma_tiler_kv[0],self.mma_tiler_kv[1]));R=G.make_fragment_C(cute.append(Q,self.tmem_kv_acc_stages));J=cute.make_tensor(tmem_ptr+self.tmem_state_offset,R.layout);S=utils.gemm.sm100.transform_partitioned_tensor_layout(J)
		if cutlass.const_expr(self.split_value_rows):K=cute.make_copy_atom(tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(16)),self.acc_dtype)
		else:K=cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),self.acc_dtype)
		T=cute.make_identity_tensor((self.mma_tiler_kv[0],self.mma_tiler_kv[1]));U=J[(_A,_A),0,0,0];L=tcgen05.make_tmem_copy(K,U);C=L.get_slice(I);V=C.partition_D(S);M=C.partition_S(T);D=cute.make_rmem_tensor_like(M,self.acc_dtype);B=cute.make_rmem_tensor_like(M,self.state_dtype)
		if cutlass.const_expr(E is not _A):N=E[F]
		else:N=F
		W=cute.flat_divide(mS_init[_A,_A,head_idx,N],(self.mma_tiler_kv[0],self.mma_tiler_kv[1]))[_A,_A,0,0];X=C.partition_S(W);O=H.acquire_and_advance()
		for A in cutlass.range(B.shape[2]):
			cute.autovec_copy(X[_A,0,A],B[_A,0,A],l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)
			if cutlass.const_expr(self.acc_dtype!=self.state_dtype):D[_A,0,A].store(B[_A,0,A].load().to(self.acc_dtype))
			else:D=B
			cute.copy(L,D[_A,0,A],V[_A,0,A,O.index])
		cute.arch.fence_view_async_tmem_store();self.init_state_store_barrier.arrive_and_wait()
		if I==0:cute.arch.mbarrier_arrive(O.barrier)
		return H
	@cute.jit
	def _store_empty_final_state(self,tidx,mS_init,mS_out,mS_indices,head_idx,batch_idx):
		E=batch_idx;D=head_idx;C=mS_indices;F=self.threads_per_warp*len(self.compute_group_1_warp_ids);I=tidx%F;J=self.mma_tiler_kv[0]*self.mma_tiler_kv[1]
		if cutlass.const_expr(C is not _A):A=C[E]
		else:A=E
		for G in cutlass.range(I,J,F,unroll=1):B=G//self.mma_tiler_kv[0];H=G-B*self.mma_tiler_kv[0];mS_out[H,B,D,A]=mS_init[H,B,D,A]
	@cute.jit
	def _store_final_state(self,tidx,mS_out,mS_indices,head_idx,batch_idx,tmem_ptr,tiled_mma_kv,kv_acc_consumer,seqlen_b,mS_checkpoints,checkpoint_offset,checkpoint_every_n_tokens):
		J=seqlen_b;I=kv_acc_consumer;H=tiled_mma_kv;G=batch_idx;F=head_idx;E=mS_indices;Q=self.threads_per_warp*len(self.compute_group_1_warp_ids);R=tidx%Q;S=H.partition_shape_C((self.mma_tiler_kv[0],self.mma_tiler_kv[1]));T=H.make_fragment_C(cute.append(S,self.tmem_kv_acc_stages));K=cute.make_tensor(tmem_ptr+self.tmem_state_offset,T.layout);U=utils.gemm.sm100.transform_partitioned_tensor_layout(K);V=cute.make_identity_tensor((self.mma_tiler_kv[0],self.mma_tiler_kv[1]))
		if cutlass.const_expr(self.split_value_rows):L=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(16)),self.acc_dtype)
		else:L=cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),self.acc_dtype)
		W=K[(_A,_A),0,0,0];M=tcgen05.make_tmem_copy(L,W);B=M.get_slice(R);X=B.partition_S(U);N=B.partition_D(V);C=cute.make_rmem_tensor_like(N,self.acc_dtype);D=cute.make_rmem_tensor_like(N,self.state_dtype);O=I.wait_and_advance()
		for A in cutlass.range(C.shape[2]):
			cute.copy(M,X[_A,0,A,O.index],C[_A,0,A])
			if cutlass.const_expr(self.acc_dtype!=self.state_dtype):D[_A,0,A].store(C[_A,0,A].load().to(self.state_dtype))
			else:D=C
			if cutlass.const_expr(self.enable_checkpoints):
				if J%checkpoint_every_n_tokens==0:
					Y=cute.ceil_div(J,self.b_t)
					if Y%2==0:Z=cute.flat_divide(mS_checkpoints[_A,_A,F,checkpoint_offset],(self.mma_tiler_kv[0],self.mma_tiler_kv[1]))[_A,_A,0,0];a=B.partition_D(Z);cute.autovec_copy(D[_A,0,A],a[_A,0,A],l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)
			if cutlass.const_expr(self.store_final_state):
				if cutlass.const_expr(E is not _A):P=E[G]
				else:P=G
				b=cute.flat_divide(mS_out[_A,_A,F,P],(self.mma_tiler_kv[0],self.mma_tiler_kv[1]))[_A,_A,0,0];c=B.partition_D(b);cute.autovec_copy(D[_A,0,A],c[_A,0,A],l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)
		O.release();return I
	@cute.jit
	def compute_group_1_chunk(self,tidx:cutlass.Int32,tmem_ptr:cutlass.Int64,scale:cutlass.Float32,mma_args:tuple,smem_args:tuple,checkpoint_args:tuple,pipeline_args:tuple,work_args:tuple)->tuple[pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineConsumer,pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer,pipeline.PipelineProducer,cutlass.Int32]:
		K=tmem_ptr;Ah,Z,x,Bk,Ai=smem_args;Aj,a,y=checkpoint_args;z,b,c,d,e,f,O,P,Q,g,A0=pipeline_args;A1,R,S=mma_args;T,Bl,Ak,A2,Al=work_args;Am=self.threads_per_warp*len(self.compute_group_1_warp_ids);D=tidx%Am;An=A1.partition_shape_C((self.mma_tiler_kv[0],self.mma_tiler_kv[1]));Ao=A1.make_fragment_C(cute.append(An,self.tmem_kv_acc_stages));A3=cute.make_tensor(K+self.tmem_state_offset,Ao.layout);A4=utils.gemm.sm100.transform_partitioned_tensor_layout(A3);Ap=cute.make_identity_tensor((self.mma_tiler_kv[0],self.mma_tiler_kv[1]))
		if cutlass.const_expr(self.split_value_rows):A5=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(16)),self.acc_dtype);A6=cute.make_copy_atom(tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(16)),self.acc_dtype)
		else:A5=cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),self.acc_dtype);A6=cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),self.acc_dtype)
		A7=A3[(_A,_A),0,0,0];h=tcgen05.make_tmem_copy(A5,A7);i=tcgen05.make_tmem_copy(A6,A7);j=h.get_slice(D);Aq=i.get_slice(D);A8=j.partition_S(A4);A9=j.partition_D(Ap);AA=Aq.partition_D(A4);B=cute.make_rmem_tensor_like(A9,self.acc_dtype);k=cute.make_rmem_tensor_like(A9,self.state_dtype);Ar=R.partition_shape_A((self.mma_tiler_qs[0],self.mma_tiler_qs[2]));As=R.make_fragment_A(cute.append(Ar,self.tmem_state_inp_stages));At=cute.make_tensor(cute.recast_ptr(K+self.tmem_state_inp_offset,dtype=self.io_dtype),As.layout);AB=utils.gemm.sm100.transform_partitioned_tensor_layout(At);Au=cute.make_identity_tensor((self.mma_tiler_qs[0],self.mma_tiler_qs[2]))
		if cutlass.const_expr(self.split_value_rows):AC=cute.make_copy_atom(tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(16)),self.io_dtype)
		else:AC=cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),self.io_dtype)
		Av=AB[_A,_A,0];l=tcgen05.make_tmem_copy(AC,Av);AD=l.get_slice(D);Aw=AD.partition_S(Au);AE=AD.partition_D(AB);U=cute.make_rmem_tensor_like(Aw,self.io_dtype);Ax=S.partition_shape_C((self.mma_tiler_qkv[0],self.mma_tiler_qkv[1]));Ay=S.make_fragment_C(Ax);AF=cute.make_tensor(K+self.tmem_cg1_shared_acc_offset,cute.flat_product(Ay.layout,cute.make_layout((self.tmem_cg1_shared_acc_stages,))));AG=utils.gemm.sm100.transform_partitioned_tensor_layout(AF);AH=cute.make_identity_tensor((self.mma_tiler_qkv[0],self.mma_tiler_qkv[1]));Az=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),self.acc_dtype);AI=AF[(_A,_A),0,0,0];AJ=tcgen05.make_tmem_copy(Az,AI);m=AJ.get_slice(D);A_=m.partition_S(AG);F=m.partition_D(AH);B0=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),self.acc_dtype);AK=tcgen05.make_tmem_copy(B0,AI);AL=AK.get_slice(D);B1=AL.partition_S(AG);B2=AL.partition_D(AH);B3=S.partition_shape_A((self.mma_tiler_qkv[0],self.mma_tiler_qkv[2]));B4=S.make_fragment_A(cute.append(B3,self.tmem_shared_inp_stages));B5=cute.make_tensor(cute.recast_ptr(K+self.tmem_shared_inp_offset,dtype=self.io_dtype),B4.layout);n=utils.gemm.sm100.transform_partitioned_tensor_layout(B5);B6=cute.make_identity_tensor((self.mma_tiler_qkv[0],self.mma_tiler_qkv[2]));B7=cute.make_copy_atom(tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)),self.io_dtype);AM=n[_A,_A,0];V=tcgen05.make_tmem_copy(B7,AM);AN=V.get_slice(D);AO=AN.partition_D(n);B8=cute.make_copy_atom(tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)),self.io_dtype);AP=tcgen05.make_tmem_copy(B8,AM);B9=AP.get_slice(D);BA=B9.partition_D(n);BB=R.partition_shape_C((self.mma_tiler_qs[0],self.mma_tiler_qs[1]));BC=R.make_fragment_C(cute.append(BB,self.tmem_q_state_acc_stages));o=cute.make_tensor(K+self.tmem_q_state_offset,BC.layout);p=utils.gemm.sm100.transform_partitioned_tensor_layout(o);AQ=cute.make_identity_tensor((self.mma_tiler_qs[0],self.mma_tiler_qs[1]));BD=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),self.acc_dtype);BE=cute.make_copy_atom(tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),self.acc_dtype);AR=o[(_A,_A),0,0,0];AS=tcgen05.make_tmem_copy(BD,AR);BF=o[(_A,_A),0,0,0];AT=tcgen05.make_tmem_copy(BE,BF);AU=AS.get_slice(D);BG=AT.get_slice(D);BH=AU.partition_S(p);BI=AU.partition_D(AQ);BJ=BG.partition_D(p);L=cute.make_rmem_tensor_like(BI,self.acc_dtype);BK=AN.partition_S(B6);BL=cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4,transpose=_B),self.io_dtype);q=cute.make_tiled_copy_S(BL,V);BM=q.get_slice(D);BN=cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),self.acc_dtype);r=tcgen05.make_tmem_copy(BN,AR);AV=r.get_slice(D);BO=AV.partition_S(p);BP=AV.partition_D(AQ);BQ=cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4,transpose=_B),self.io_dtype);s=cute.make_tiled_copy_D(BQ,r);BR=s.get_slice(D);BS=BR.partition_D(Ai);BT=32;BU=utils.gemm.sm100.transform_partitioned_tensor_layout(Ah);BV=BM.partition_S(BU);BW=cute.make_rmem_tensor((1,cute.size(F)),self.acc_dtype);t=m.partition_D(BW);u=cute.make_rmem_tensor_like(F,self.acc_dtype);BX=T&1==0;W=Al==_C
		if cutlass.const_expr(self.use_initial_state):
			W=_B
			if BX:f.advance();f.advance()
		if cutlass.const_expr(self.use_ready_only_scalar):G=T%self.smem_gate_stages;BY=T//self.smem_gate_stages&1;b.wait(G,BY)
		else:AW=b.wait_and_advance();G=AW.index
		AX=x[self.b_t-1,0,G];H=d.current_handle()
		if W:
			H=d.wait_and_advance()
			if cutlass.const_expr(not self.compact_cg1_events):BZ=O.current_handle();O.advance()
			if cutlass.const_expr(self.split_value_rows):
				for E in cutlass.range(B.shape[2]):
					cute.copy(h,A8[_A,0,E,H.index],B[_A,0,E]);U[_A,0,E].store(B[_A,0,E].load().to(self.io_dtype));cute.copy(l,U[_A,0,E],AE[_A,0,E,0])
					for AY in cutlass.range(cute.size(B[_A,0,E]),vectorize=_B):B[_A,0,E][AY]=B[_A,0,E][AY]*AX
					cute.copy(i,B[_A,0,E],AA[_A,0,E,H.index])
			else:cute.copy(h,A8[_A,0,_A,H.index],B[_A,0,_A]);U[_A,0,_A].store(B[_A,0,_A].load().to(self.io_dtype));cute.copy(l,U[_A,0,_A],AE[_A,0,_A,0])
			cute.arch.fence_view_async_tmem_store()
			if cutlass.const_expr(self.compact_cg1_events):
				with cute.arch.elect_one():O.arrive(0,_A)
			else:BZ.commit()
			I=self.b_t*T
			if cutlass.const_expr(self.enable_checkpoints):
				if I>0:
					if I<=A2:
						if I%y==0:
							Ba=cute.flat_divide(Aj[_A,_A,Ak,a],(self.mma_tiler_kv[0],self.mma_tiler_kv[1]))[_A,_A,0,0];Bb=j.partition_D(Ba)
							if cutlass.const_expr(self.state_dtype!=self.acc_dtype):k[_A,0,_A].store(B[_A,0,_A].load().to(self.state_dtype))
							else:k=B
							cute.autovec_copy(k[_A,0,_A],Bb[_A,0,_A],l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)
			if cutlass.const_expr(not self.split_value_rows):
				for A in cutlass.range(cute.size(B),vectorize=_B):B[A]=B[A]*AX
				cute.copy(i,B[_A,0,_A],AA[_A,0,_A,H.index])
			if cutlass.const_expr(self.enable_checkpoints):
				if I>0:
					if I<=A2:
						if I%y==0:a+=1
			cute.arch.fence_view_async_tmem_store();H.release()
		for A in cutlass.range_constexpr(cute.size(F)):Bc=F[A];t[A]=x[Bc[1],0,G]
		AZ=Z[self.b_t-1,0,G]
		for A in cutlass.range_constexpr(0,cute.size(F),2):Bd=F[A];Be=F[A+1];Aa=cute.arch.add_packed_f32x2((AZ,AZ),(-Z[Bd[1],0,G],-Z[Be[1],0,G]),ftz=_C,rnd='rn');u[A]=cute.math.exp2(Aa[0],fastmath=_B);u[A+1]=cute.math.exp2(Aa[1],fastmath=_B)
		if cutlass.const_expr(not self.use_ready_only_scalar):AW.release()
		if cutlass.const_expr(not self.compact_cg1_events):Bf=P.current_handle();P.advance()
		Ab=z.wait_and_advance();X=cute.make_rmem_tensor_like(BK,self.io_dtype);Bg=q.retile(X);cute.copy(q,BV[_A,_A,_A,Ab.index],Bg)
		if W:
			Ac=c.wait_and_advance();J=cute.make_rmem_tensor_like(B2,self.acc_dtype);cute.copy(AK,B1[_A,_A,_A,Ac.index],J)
			for A in cutlass.range(cute.size(J),vectorize=_B):J[A]=J[A]*t[A]
			Ac.release()
			for A in cutlass.range(cute.size(J),vectorize=_B):X[A]=X[A]-J[A].to(self.io_dtype)
		cute.copy(AP,X,BA[_A,_A,_A,0]);cute.arch.fence_view_async_tmem_store()
		if cutlass.const_expr(self.compact_cg1_events):
			with cute.arch.elect_one():P.arrive(0,_A)
		else:Bf.commit()
		if W:
			v=e.wait_and_advance();cute.copy(AS,BH[_A,_A,0,v.index],L[_A,_A,0])
			for A in cutlass.range(cute.size(L),vectorize=_B):L[A]=L[A]*t[A]*scale
			cute.copy(AT,L[_A,_A,0],BJ[_A,_A,0,v.index]);cute.arch.fence_view_async_tmem_store();v.release()
		Ad=c.wait_and_advance();Ab.release();M=cute.make_rmem_tensor_like(F,self.acc_dtype);Y=cute.make_rmem_tensor_like(M,self.io_dtype)
		for C in cutlass.range(M.shape[1]):cute.copy(AJ,A_[_A,C,0,Ad.index],M[_A,C,0]);Y[_A,C,0].store(M[_A,C,0].load().to(self.io_dtype))
		Ad.release();N=M
		for C in cutlass.range(N.shape[1]):
			for A in cutlass.range(BT,vectorize=_B):N[A,C,0]=N[A,C,0]*u[A,C,0]
		if cutlass.const_expr(not self.compact_cg1_events):Bh=Q.current_handle();Q.advance();Bi=g.current_handle();g.advance()
		for C in cutlass.range(N.shape[1]):cute.copy(V,Y[_A,C,0],AO[_A,C,0,0]);Y[_A,C,0].store(N[_A,C,0].load().to(self.io_dtype));cute.copy(V,Y[_A,C,0],AO[_A,C,0,1])
		cute.arch.fence_view_async_tmem_store()
		if cutlass.const_expr(self.compact_cg1_events):
			with cute.arch.elect_one():Q.arrive(0,_A)
		else:Bh.commit();Bi.commit()
		Ae=A0.acquire_and_advance();Af=e.wait_and_advance();w=cute.make_rmem_tensor_like(BP,self.acc_dtype);Ag=cute.make_rmem_tensor_like(w,self.io_dtype);Bj=s.retile(Ag);cute.copy(r,BO[_A,_A,_A,Af.index],w);Ag.store(w.load().to(self.io_dtype));cute.copy(s,Bj,BS[_A,_A,_A,Ae.index]);cute.arch.fence_view_async_shared();Af.release();Ae.commit();return z,b,c,d,e,f,O,P,Q,g,A0,a
	@cute.jit
	def epilogue_warp(self,smem_args,tma_args,pipeline_args,work_args,tensormap_args)->pipeline.PipelineConsumer:
		F,=smem_args;A,=tma_args;C,=pipeline_args;G,H=work_args;I,J=tensormap_args;B=C.wait_and_advance();K=cute.make_layout(1);L=cute.select(self.mma_tiler_qkv,mode=[0,1]);M=cute.domain_offset((cutlass.Int32(0),H),A.tma_tensor[_A,_A,G]);N=cute.flat_divide(M,L);D,E=cpasync.tma_partition(A.atom,0,K,cute.group_modes(F,0,2),cute.group_modes(N,0,2))
		if cutlass.const_expr(self.use_static_tma_descriptors):cute.copy(A.atom,D[_A,B.index],E[_A,0,0])
		else:cute.copy(A.atom,D[_A,B.index],E[_A,0,0],tma_desc_ptr=I.get_tensormap_ptr(J,cute.AddressSpace.generic))
		cute.arch.cp_async_bulk_commit_group();cute.arch.cp_async_bulk_wait_group(0);B.release();return C
	@cute.jit
	def _transform_to_position_independent_layout(self,tensor:cute.Tensor,swizzle_inner:cute.Swizzle)->cute.Tensor:B=swizzle_inner;A=tensor;C=cute.recast_ptr(A.iterator,swizzle_=_A);D=int(math.log2(self.io_dtype.width))-1;E=cute.make_swizzle(B.num_bits,D,B.num_shift);F=cute.make_composed_layout(E,0,A.layout);return cute.make_tensor(C,F)
	@staticmethod
	def get_workspace_size(num_sm:int,B:int,HQ:int,HV:int,is_persistent:bool):
		if is_persistent:return GatedDeltaNetChunkedKernel.bytes_per_tensormap*GatedDeltaNetChunkedKernel.num_tensormaps*num_sm
		A=HQ if HQ>=HV else HV;return GatedDeltaNetChunkedKernel.bytes_per_tensormap*GatedDeltaNetChunkedKernel.num_tensormaps*(B*A)
	@cute.jit
	def initialize_workspace(self,workspace:cute.Tensor,grid_dim:Tuple[int,int,int]):B=grid_dim;A=workspace;A=cute.make_tensor(A.iterator,cute.make_layout((B[0]*B[1]*B[2],GatedDeltaNetChunkedKernel.num_tensormaps,GatedDeltaNetChunkedKernel.bytes_per_tensormap),stride=(GatedDeltaNetChunkedKernel.num_tensormaps*GatedDeltaNetChunkedKernel.bytes_per_tensormap,GatedDeltaNetChunkedKernel.bytes_per_tensormap,1)));return A
import functools
from typing import Optional
import torch,cuda.bindings.driver as cuda,cutlass,cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
@functools.cache
def get_num_sm(device:torch.device)->int:return torch.cuda.get_device_properties(device).multi_processor_count
@functools.cache
def _get_compiled_cache(io_dtype_str:str,state_dtype_str:str,HQ:int,HV:int,is_GQA:bool,use_initial_state:bool,store_final_state:bool,enable_checkpoints:bool,use_state_indices:bool,use_m64_rows:bool,use_static_tma_descriptors:bool,static_tma_full_chunks:bool,use_static_scale:bool,compact_cg1_events:bool):return{}
def _cutlass_io_dtype(torch_dtype:torch.dtype):
	A=torch_dtype
	if A==torch.bfloat16:return cutlass.BFloat16
	elif A==torch.float16:return cutlass.Float16
	else:raise ValueError(f"Unsupported dtype {A}, expected bfloat16 or float16")
def _cutlass_state_dtype(torch_dtype:torch.dtype):
	A=torch_dtype
	if A==torch.float32:return cutlass.Float32
	elif A==torch.bfloat16:return cutlass.BFloat16
	elif A==torch.float16:return cutlass.Float16
	elif A==torch.float8_e4m3fn:return cutlass.Float8E4M3FN
	elif A==torch.float8_e5m2:return cutlass.Float8E5M2
	else:raise ValueError(f"Unsupported state dtype {A}, expected float32, bfloat16, float16, float8_e4m3fn, or float8_e5m2")
@functools.cache
def _device_capability(device_index:int)->tuple[int,int]:return torch.cuda.get_device_capability(device_index)
def _device_index(q:torch.Tensor)->int:return q.device.index if q.device.index is not _A else torch.cuda.current_device()
def get_gdn_launch_metadata(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,gate:torch.Tensor,beta:torch.Tensor,output:torch.Tensor,cu_seqlens:torch.Tensor,initial_state:Optional[torch.Tensor],output_state:Optional[torch.Tensor],scale:float,checkpoint_every_n_tokens:int=0,state_indices:Optional[torch.Tensor]=_A,*,value_rows_override:int|_A=_A)->dict[str,object]:
	I=output;C=value_rows_override;B=output_state;A=initial_state
	if C not in(_A,64,128):raise ValueError('value_rows_override must be None, 64, or 128')
	if q.dtype not in(torch.float16,torch.bfloat16):raise ValueError('q must be float16 or bfloat16')
	if k.dtype!=q.dtype or v.dtype!=q.dtype or I.dtype!=q.dtype:raise ValueError('k, v, and output must have the same dtype as q')
	if gate.dtype!=torch.float32 or beta.dtype!=torch.float32:raise ValueError('gate and beta must be float32')
	M=_device_index(q);J=_device_capability(M);F=q.size(1);G=v.size(1);K=q.size(2);L=cu_seqlens.size(0)-1;N=checkpoint_every_n_tokens>0;O=state_indices is not _A;H=J==(10,3)and K==128 and k.size(2)==128 and v.size(2)==128 and G>=F and G%F==0 and A is not _A and B is not _A and A.dtype==torch.float32 and B.dtype==torch.float32 and not N and not O
	if C==64 and not H:raise ValueError('value_rows_override=64 requires natural M64 eligibility')
	D=H and C!=128;E=D and L==1;P=E and q.size(0)%128==0;Q=E and scale==K**-.5;R=E or D and L==2 and F==G;return{'requested_value_rows':C,'selected_value_rows':64 if D else 128,'capability':list(J),'dtypes':{'q':str(q.dtype),'k':str(k.dtype),'v':str(v.dtype),'gate':str(gate.dtype),'beta':str(beta.dtype),'output':str(I.dtype),'initial_state':str(A.dtype)if A is not _A else _A,'output_state':str(B.dtype)if B is not _A else _A},'natural_m64_eligible':H,_F:D,_G:E,_H:P,_I:Q,_J:R}
def _mark_state_layout(s_cute,use_state_indices:bool,DK:int)->_A:
	A=s_cute
	if use_state_indices:A.mark_layout_dynamic()
	else:A.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3,stride_order=(0,1,2,3),divisibility=DK)
def chunk_gated_delta_rule_sm100(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,gate:torch.Tensor,beta:torch.Tensor,output:torch.Tensor,cu_seqlens:torch.Tensor,initial_state:Optional[torch.Tensor],output_state:Optional[torch.Tensor],scale:float,checkpoint_every_n_tokens:int=0,cu_checkpoints:Optional[torch.Tensor]=_A,output_checkpoints:Optional[torch.Tensor]=_A,state_indices:Optional[torch.Tensor]=_A,*,value_rows_override:int|_A=_A)->_A:
	A2='num_sm';f=output_checkpoints;e=cu_checkpoints;d='compiled';T=state_indices;S=scale;R=output;Q=beta;P=gate;H=checkpoint_every_n_tokens;G=cu_seqlens;D=output_state;C=initial_state;I=q.size(1);J=v.size(1);U=q.size(2);g=I>=J;K=C is not _A;L=D is not _A;V=H>0;h=_cutlass_io_dtype(q.dtype)
	if C is not _A:M=C.dtype
	elif D is not _A:M=D.dtype
	else:M=torch.float32
	A3=_cutlass_state_dtype(M);i=C if K else _A;j=G.size(0)-1;l=D if L else _A;E=T is not _A;m=T if E else _A;F=get_gdn_launch_metadata(q=q,k=k,v=v,gate=P,beta=Q,output=R,cu_seqlens=G,initial_state=C,output_state=D,scale=S,checkpoint_every_n_tokens=H,state_indices=T,value_rows_override=value_rows_override);n=bool(F[_F]);o=bool(F[_G]);p=bool(F[_H]);r=bool(F[_I]);s=bool(F[_J]);A=_get_compiled_cache(str(q.dtype),str(M),I,J,g,K,L,V,E,n,o,p,r,s)
	if d not in A:
		B=get_num_sm(q.device);A4=B;W=64 if n else 128;A5=GatedDeltaNetChunkedKernel(io_dtype=h,inverse_dtype=h,acc_dtype=cutlass.Float32,state_dtype=A3,mma_tiler_qk=(64,64,128),mma_tiler_qs=(W,64,128),mma_tiler_qkv=(W,64,64),mma_tiler_kv=(W,128,64),max_active_clusters=A4,num_sm=B,is_GQA=g,use_initial_state=K,store_final_state=L,enable_checkpoints=V,is_persistent=_B,use_static_tma_descriptors=o,static_tma_full_chunks=p,use_static_scale=r,compact_cg1_events=s);t=from_dlpack(q,assumed_align=16);t.mark_compact_shape_dynamic(mode=0,stride_order=(0,1,2),divisibility=1);u=from_dlpack(k,assumed_align=16);u.mark_compact_shape_dynamic(mode=0,stride_order=(0,1,2),divisibility=1);w=from_dlpack(v,assumed_align=16);w.mark_compact_shape_dynamic(mode=0,stride_order=(0,1,2),divisibility=1);x=from_dlpack(P,assumed_align=16);x.mark_compact_shape_dynamic(mode=0,stride_order=(0,1),divisibility=1);y=from_dlpack(Q,assumed_align=16);y.mark_compact_shape_dynamic(mode=0,stride_order=(0,1),divisibility=1);z=from_dlpack(R,assumed_align=16);z.mark_compact_shape_dynamic(mode=0,stride_order=(0,1,2),divisibility=1);A6=from_dlpack(G,assumed_align=4).mark_layout_dynamic();X=_A
		if K:X=from_dlpack(i,assumed_align=16);_mark_state_layout(X,E,U)
		Y=_A
		if L:Y=from_dlpack(l,assumed_align=16);_mark_state_layout(Y,E,U)
		A0=_A
		if E:A0=from_dlpack(m,assumed_align=4).mark_layout_dynamic()
		Z=_A;A1=_A
		if V:Z=from_dlpack(f,assumed_align=16);Z.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3,stride_order=(0,1,2,3),divisibility=U);A1=from_dlpack(e,assumed_align=4).mark_layout_dynamic()
		N=GatedDeltaNetChunkedKernel.get_workspace_size(B,j,I,J,_B);a=torch.empty(N,dtype=torch.int8,device=q.device);A7=from_dlpack(a,assumed_align=16);b=cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream);c=cute.compile(A5,t,u,w,x,y,z,A6,X,Y,A0,Z,A1,H,S,A7,b,options='--enable-tvm-ffi --opt-level 3');A[d]=c;A[A2]=B
	c=A[d];B=A[A2];N=GatedDeltaNetChunkedKernel.get_workspace_size(B,j,I,J,_B);O=f"workspace_{q.device.index}"
	if O not in A or A[O].size(0)<N:A[O]=torch.empty(N,dtype=torch.int8,device=q.device)
	a=A[O];b=cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream);c(q,k,v,P,Q,R,G,i,l,m,f,e,H,S,a,b)
class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		if torch.cuda.get_device_capability()!=(10,3):raise RuntimeError('This solution targets NVIDIA L20D / SM103')
		if torch.version.cuda is _A or int(torch.version.cuda.split('.')[0])<13:raise RuntimeError('The SM103 GDN kernel requires CUDA 13+')
	@torch.inference_mode()
	def forward(self,q,k,v,alpha,beta,initial_state,cu_seqlens,scale,output,output_state):B=output_state;A=output;chunk_gated_delta_rule_sm100(q,k,v,alpha,beta,A,cu_seqlens.to(torch.int32),initial_state,B,scale,value_rows_override=64);return A,B
