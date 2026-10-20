_B=1.
_A=None
import torch,torch.nn.functional as F
def exclusive_cumsum(a:list[int]):
	A=[0]
	for B in a:A.append(A[-1]+B)
	return A
def matmul(a:torch.Tensor,b:torch.Tensor):
	assert a.dtype in[torch.float16,torch.bfloat16,torch.float32,torch.float64];assert b.dtype in[torch.float16,torch.bfloat16,torch.float32,torch.float64]
	if a.dtype==torch.float16 or b.dtype==torch.float16 or a.dtype==torch.bfloat16 or b.dtype==torch.bfloat16:
		B=a.to(torch.float32);C=b.to(torch.float32);A=B@C
		if a.dtype==torch.bfloat16:return A
		else:return A.to(torch.float16)
	else:return a@b
def _linear_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,*,decay_factor:torch.Tensor|_A=_A,qk_weight:torch.Tensor|_A=_A)->torch.Tensor:
	E=qk_weight;C=decay_factor;F=q.shape[1];H=k.shape[1];assert F==H;q=q.transpose(0,1);k=k.transpose(0,1);v=v.transpose(0,1);I=matmul(q,k.transpose(-2,-1));G=q.size(-2);B=torch.tril(torch.ones(F,G,G,dtype=q.dtype,device=q.device))
	if C is not _A and(C!=_B).any():
		N,J,K=B.shape
		with torch.device(q.device):L=(torch.arange(J).unsqueeze(1)-torch.arange(K).unsqueeze(0)).unsqueeze(0);A=torch.pow(C,L);A[B==.0]=.0
	elif E is not _A:A=E.clone();A[B==.0]=.0
	else:A=B
	M=I*A;D=matmul(M,v);D=D.transpose(0,1);return D
def identity_add_strict_lower_diagonal(m:torch.Tensor):
	A=m.size(-1);assert m.size(-2)==A
	with torch.device(m.device):m=m.clone();B=torch.arange(A).unsqueeze(1)<=torch.arange(A);m[:,B]=.0;m=m+torch.eye(A).unsqueeze(0)
	return m
def to_logspace_Gamma_and_gamma(alpha_HS:torch.Tensor,epsilon=1e-10):B=torch.log(alpha_HS+epsilon);A=torch.cumsum(B,dim=-1);C=A.unsqueeze(2)-A.unsqueeze(1);D=A.unsqueeze(2);return C,D
@torch.inference_mode
def blockwise_delta_rule(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,seq_lens:list[int],alpha:torch.Tensor|_A=_A,beta:torch.Tensor|_A=_A,block_size:int=32,scale_factor=_B,state_dtype:torch.dtype=torch.float32,initial_state:torch.Tensor|_A=_A)->torch.Tensor:
	b=scale_factor;a=seq_lens;T=initial_state;S=state_dtype;E=beta;D=alpha;B=block_size;c=q.size(0);F=q.size(1);d=k.size(1);G=v.size(1);I=max(F,G);J=q.size(2)
	if D is _A:D=torch.ones(c,I,dtype=torch.float32,device=q.device)
	if E is _A:E=torch.ones(c,I,dtype=torch.float32,device=q.device)
	if F>G:M=F;k=k.repeat_interleave(F//d,dim=1);v=v.repeat_interleave(F//G,dim=1)
	else:M=G;q=q.repeat_interleave(G//F,dim=1);k=k.repeat_interleave(G//d,dim=1)
	K=torch.zeros((len(a),I,J,J),dtype=S,device=q.device)
	if T is not _A:assert tuple(T.shape)==tuple(K.shape);K.copy_(T.to(S))
	e=torch.zeros_like(q);f=exclusive_cumsum(a)
	for(U,o)in enumerate(f[:-1]):
		C=f[U+1];A=o;L=K[U].clone()
		while A<C:
			V=C-A>=B;H=B if V else C-A;W=e[A:min(C,A+B)]
			if V:N=q[A:A+B];O=k[A:A+B];X=v[A:A+B];Y=D[A:A+B];Z=E[A:A+B]
			else:N=torch.zeros((B,M,J),dtype=q.dtype,device=q.device);O=torch.zeros((B,M,J),dtype=k.dtype,device=k.device);X=torch.zeros((B,M,J),dtype=v.dtype,device=v.device);Y=torch.ones(B,I,dtype=D.dtype,device=D.device);Z=torch.zeros(B,I,dtype=E.dtype,device=E.device);N[:H]=q[A:C];O[:H]=k[A:C];X[:H]=v[A:C];Y[:H]=D[A:C];Z[:H]=E[A:C]
			p=Y.transpose(0,1);g=Z.transpose(0,1).unsqueeze(2);h,P=to_logspace_Gamma_and_gamma(p);i=P[:,[H-1],:];r=N.transpose(0,1);Q=O.transpose(0,1);s=X.transpose(0,1);t=identity_add_strict_lower_diagonal(g*torch.exp(h)*matmul(Q,Q.transpose(-2,-1)));R=torch.inverse(t)*g.transpose(1,2);R=R.to(q.dtype);j=matmul(R,s);u=matmul(R,torch.exp(P)*Q);l=j-matmul(u.to(torch.float32),L.to(torch.float32)).to(j.dtype);w=l.transpose(0,1);m=matmul(torch.exp(P)*r.to(torch.float32),L.to(torch.float32)).transpose(0,1).to(q.dtype);n=_linear_attention(N,O,w,qk_weight=torch.exp(h))
			if V:W[:]=b*(m+n)
			else:W[:]=b*(m+n)[:W.shape[0]]
			x=matmul((torch.exp(i-P)*Q).transpose(-2,-1).to(torch.float32),l.to(torch.float32));L=(torch.exp(i)*L.to(torch.float32)+x).to(S);A+=B
		K[U,:,:,:]=L
	return e,K
import functools,math,torch
def multidist_randu(num_dists,dim,mean_mean=.0,mean_std=_B,lower=-_B,upper=_B):A=torch.distributions.Normal(mean_mean,mean_std).sample((num_dists,));B=torch.distributions.Uniform(A+lower,A+upper).sample((dim,));return B.T.contiguous()
def gen_qkv(seq_lens,num_q_heads,num_k_heads,num_v_heads,head_size,dtype=torch.float16):J=num_v_heads;I=num_k_heads;H=num_q_heads;C=dtype;A=head_size;D=functools.partial(multidist_randu,mean_std=.05,lower=-.25,upper=.25);B=sum(seq_lens);E=D(B*H,A);F=D(B*I,A);G=D(B*J,A);E=E.reshape(B,H,A).to(C).contiguous();F=F.reshape(B,I,A).to(C).contiguous();G=G.reshape(B,J,A).to(C).contiguous();return E,F,G
def _make_inputs(sequence_lengths,initial_state_mode='zero',device='cuda'):
	J='nan';I='continuation';H=initial_state_mode;A=sequence_lengths
	if not isinstance(A,list)or not 1<=len(A)<=16:raise ValueError('sequence_lengths must contain 1..16 sequences')
	if any(type(A)is not int or A<1 for A in A):raise ValueError('each sequence length must be a positive integer')
	B=sum(A)
	if B>16384:raise ValueError('sum(sequence_lengths) must be <= 16384')
	if H not in{'zero',I}:raise ValueError('initial_state_mode must be zero or continuation')
	with torch.device(device):
		K,D,L=gen_qkv(A,4,4,32,128,torch.bfloat16);D=torch.nn.functional.normalize(D,p=2.,dim=-1);M=torch.rand(B,32,dtype=torch.float32);N=torch.rand(B,32,dtype=torch.float32);E=torch.zeros(len(A),32,128,128,dtype=torch.float32)
		if H==I:C=[32]*len(A);O,F,P=gen_qkv(C,4,4,32,128,torch.bfloat16);F=torch.nn.functional.normalize(F,p=2.,dim=-1);Q=torch.rand(sum(C),32,dtype=torch.float32);R=torch.rand(sum(C),32,dtype=torch.float32);X,S=blockwise_delta_rule(O.float(),F.float(),P.float(),C,alpha=Q,beta=R,scale_factor=_B/math.sqrt(128),state_dtype=torch.float32);E=S.transpose(-1,-2).contiguous()
		G=[0]
		for T in A:G.append(G[-1]+T)
		U=torch.tensor(G,dtype=torch.int64);V=torch.full((B,32,128),float(J),dtype=torch.bfloat16);W=torch.full_like(E,float(J))
	return dict(q=K,k=D,v=L,alpha=M,beta=N,initial_state=E,cu_seqlens=U,output=V,output_state=W,scale=_B/math.sqrt(128))
