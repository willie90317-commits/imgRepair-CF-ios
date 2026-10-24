"""Core ML friendly re-expression of CodeFormer.forward (inference path only).

Same math as the original with adain=True, rewritten so every op converts cleanly:
- softmax + topk + one-hot matmul codebook lookup  ->  argmax + gather (softmax is monotonic)
- nn.MultiheadAttention                            ->  explicit q/k/v attention
- `if w > 0` Python branch                         ->  always fuse, scaled by the `fidelity`
  input (w = 0 gives an all-zero residual, identical to skipping the fusion)
- input  : RGB normalised to [-1, 1]  (Core ML ImageType does this with scale/bias)
- output : RGB in [0, 255]            (so it can be returned as an image)
"""
import torch
import torch.nn.functional as F
from torch import nn


def _mean_std(feat, eps=1e-5):
    n = feat.shape[2] * feat.shape[3]
    mean = feat.mean(dim=(2, 3), keepdim=True)
    var = ((feat - mean) ** 2).sum(dim=(2, 3), keepdim=True) / (n - 1) + eps  # unbiased, like torch.var
    return mean, var.sqrt()


def _adain(content, style):
    s_mean, s_std = _mean_std(style)
    c_mean, c_std = _mean_std(content)
    return (content - c_mean) / c_std * s_std + s_mean


def _self_attention_layer(layer, tgt, pos):
    t2 = layer.norm1(tgt)
    qk = t2 + pos
    attn = layer.self_attn
    e, h = attn.embed_dim, attn.num_heads
    d = e // h
    w, b = attn.in_proj_weight, attn.in_proj_bias
    q = F.linear(qk, w[:e], b[:e])
    k = F.linear(qk, w[e:2 * e], b[e:2 * e])
    v = F.linear(t2, w[2 * e:], b[2 * e:])
    length = q.shape[0]
    q = q.reshape(length, h, d).permute(1, 0, 2) * (d ** -0.5)
    k = k.reshape(length, h, d).permute(1, 0, 2)
    v = v.reshape(length, h, d).permute(1, 0, 2)
    a = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)
    o = torch.matmul(a, v).permute(1, 0, 2).reshape(length, 1, e)
    tgt = tgt + attn.out_proj(o)
    t2 = layer.norm2(tgt)
    return tgt + layer.linear2(layer.activation(layer.linear1(t2)))


class ExportCodeFormer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, fidelity):
        net = self.net
        x = image
        # Block index -> feature size key; fixed by the architecture, so no shape-based lookups
        # (shapes become tensors while tracing).
        enc_ids = {net.fuse_encoder_block[s]: s for s in net.connect_list}
        gen_ids = {net.fuse_generator_block[s]: s for s in net.connect_list}
        enc = {}
        for i, block in enumerate(net.encoder.blocks):
            x = block(x)
            if i in enc_ids:
                enc[enc_ids[i]] = x
        lq = x                                                   # 1 x 256 x 16 x 16

        pos = net.position_emb.unsqueeze(1)                      # 256 x 1 x 512
        query = net.feat_emb(lq.flatten(2).permute(2, 0, 1))     # 256 x 1 x 512
        for layer in net.ft_layers:
            query = _self_attention_layer(layer, query, pos)
        logits = net.idx_pred_layer(query)                       # 256 x 1 x 1024
        idx = torch.argmax(logits[:, 0, :], dim=-1)              # 256
        quant = net.quantize.embedding.weight[idx]               # 256 x 256
        quant = quant.reshape(1, 16, 16, 256).permute(0, 3, 1, 2)
        x = _adain(quant, lq)

        w = fidelity.reshape(1, 1, 1, 1)
        for i, block in enumerate(net.generator.blocks):
            x = block(x)
            if i in gen_ids:
                key = gen_ids[i]
                fuse = net.fuse_convs_dict[key]
                feat = fuse.encode_enc(torch.cat([enc[key], x], dim=1))
                x = x + w * (x * fuse.scale(feat) + fuse.shift(feat))
        return (x.clamp(-1, 1) + 1) * 127.5
