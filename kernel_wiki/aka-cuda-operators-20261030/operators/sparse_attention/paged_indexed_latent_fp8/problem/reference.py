"""Independent FP32 attention from exactly supplied E4M3 values and scalar scales."""
import torch

def evaluate(query,kv_cache,block_tables,seq_lens,sparse_mla_top_k_lens,
             bmm1_scale,bmm2_scale,workspace_buffer,multi_ctas_kv_counter_buffer,out):
    result=torch.empty_like(out);flat=kv_cache.reshape(-1,kv_cache.shape[-1])
    for r in range(len(query)):
        n=int(seq_lens[r])
        if sparse_mla_top_k_lens is not None:n=min(n,int(sparse_mla_top_k_lens[r]))
        ids=block_tables[r,0,:n];ids=ids[ids>=0].long()
        if not len(ids):result[r]=0;continue
        stored=flat[ids].float()
        scores=query[r,0].float()@stored.T*bmm1_scale
        result[r,0]=(torch.softmax(scores,-1)@(stored[:,:512]*bmm2_scale)).to(out.dtype)
    return [result]


class Model(torch.nn.Module):
    def forward(self, query, kv_cache, block_tables, seq_lens, sparse_mla_top_k_lens, bmm1_scale, bmm2_scale, workspace_buffer, multi_ctas_kv_counter_buffer, out):
        torch.backends.cuda.matmul.allow_tf32=False
        values=evaluate(query, kv_cache, block_tables, seq_lens, sparse_mla_top_k_lens, bmm1_scale, bmm2_scale, workspace_buffer, multi_ctas_kv_counter_buffer, out)
        out.copy_(values[0])
        return out
