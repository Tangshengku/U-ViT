import torch
import torch.nn as nn
import math
import numpy as np
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
from torchvision.utils import save_image

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')

def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts

certainty = []
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class LTE(nn.Module):
    def __init__(self, embed_dim, patch_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_dim = patch_dim
        self.lte = nn.Linear(self.embed_dim, self.patch_dim)
        self.actn = nn.Sigmoid()
    def forward(self, x):
        return self.actn(self.lte(x))

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        # self.lte_classifer = nn.Linear(embed_dim * num_patches, 2)
        # self.lte_actn = nn.Softmax(dim=1)
        self.lte_classifer = nn.ModuleList([
            LTE(embed_dim, self.patch_dim) for _ in range(depth + 1)
        ])
        # self.local_lte = nn.Linear(self.embed_dim, self.patch_dim)
        # # self.quality_cov = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1)
        # self.local_lte_actn = nn.Sigmoid()
        # for _ in range(depth + 1):
        #     self.lte_classifer.append(nn.Linear(embed_dim, self.patch_dim))
        # self.lte_actn = nn.Sigmoid()
        
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        # self.freeze_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def freeze_backbone(self):
        for (name, param) in self.named_parameters():
            if "local" in name:
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False

    def output_forward(self, x, L):
        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x

    def lte(self, x, L, layer, save_uncertanty_figure=False):
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :].float().detach()
        x =  self.lte_classifer[layer](x)
        x = unpatchify(x, self.in_chans)
        # save_uncertanty_figure = True
        if save_uncertanty_figure:
            path = "/home/dongk/dkgroup/tsk/projects/U-ViT/workdir/celeba64_uvit_small/deediff/uncertainty/"
            import os
            i = len(os.listdir(path))
            # mask1 = x > 0.5
            # mask2 = x < 0.5
            x = x.mean(dim=1)
            # mask = torch.ones_like(x)
            # mask[x < 0.8] = 0
            save_image(x, path + "{}.png".format(i))
        return x
    
    # def local_lte_fn(self, x, L, layer, save_uncertanty_figure=False):
    #     assert x.size(1) == self.extras + L
    #     x = x[:, self.extras:, :].float().detach()
    #     x =  self.local_lte_actn(self.local_lte(x))
    #     x = unpatchify(x, self.in_chans)
    #     # save_uncertanty_figure = True
    #     if save_uncertanty_figure:
    #         path = "/home/dongk/dkgroup/tsk/projects/U-ViT/workdir/celeba64_uvit_small/deediff/uncertainty/"
    #         import os
    #         i = len(os.listdir(path))
    #         # mask1 = x > 0.5
    #         # mask2 = x < 0.5
    #         x = x.mean(dim=1)
    #         # mask = torch.ones_like(x)
    #         # mask[x < 0.8] = 0
    #         save_image(x, path + "{}.png".format(i))
    #     return x

    def forward(self, x, timesteps, y=None, is_train=True, layer=13, thres=0.42, nsr=None, cum_alpha=None):
        # x_clone = x.clone()
        x = self.patch_embed(x)
        
        B, L, D = x.shape
        # j = torch.tensor([0.0], device=x.device)
        j = 0
        lte_val = 0
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        inner_state = []
        lte = []
        local_lte = []
        for i, blk in enumerate(self.in_blocks):
            x = blk(x)
            skips.append(x)
            inner_state.append(self.output_forward(x, L))
            # inner_state.append(x.clone().contiguous())
            lte_val = self.lte(x, L, i)
            lte.append(lte_val)
            # local_lte.append(self.local_lte_fn(x, L, i))
            # if i == 0:
            #     global certainty
            #     certainty.append(torch.cosine_similarity(x, ))
            #     print(certainty)
                # print(torch.mean(lte_val.view(-1)).item())
            # if i == layer - 1:
            #     return self.output_forward(x, L), inner_state, local_lte, lte, j+1
            # print("layer {}: ".format(i+1), torch.mean(lte_val.view(-1)))
            if not is_train:
                # print(torch.mean(lte_val.view(-1)))
                if torch.mean(lte_val.view(-1)) > 0.95:
                    j += i 
                    with open("layer.txt", "a") as f:
                        f.write(str(j+1) + "\n")
                    # print("return at in: {}".format(i))
                    return self.output_forward(x, L), inner_state, lte, j+1
        j += i + 1
        x = self.mid_block(x)
        inner_state.append(self.output_forward(x, L))
        # inner_state.append(x.clone().contiguous())
        lte_val = self.lte(x, L, 6)
        lte.append(lte_val)
        # local_lte.append(self.local_lte_fn(x, L, 6))

        # print("layer {}: ".format(j+1), torch.mean(lte_val.view(-1)))
        if not is_train:
                # print(torch.mean(lte_val.view(-1)))
                if torch.mean(lte_val.view(-1)) > 0.9:
                    # print("return at mid")
                    j += 1 
                    with open("layer.txt", "a") as f:
                        f.write(str(j+1) + "\n")
                    return self.output_forward(x, L), inner_state, lte, j+1
        j += 1
        for i, blk in enumerate(self.out_blocks):
            x = blk(x, skips.pop())
            # if i == len(self.out_blocks) - 2:
            #     self.lte(x, L, save_uncertanty_figure=True)
            # if i == len(self.out_blocks) - 1:
            #     break
            inner_state.append(self.output_forward(x, L))
            # inner_state.append(x.clone().contiguous())
            lte_val = self.lte(x, L, i + 7)
            lte.append(lte_val)
            # if i==4:
            #     global certainty
            #     certainty.append(torch.mean(self.local_lte_fn(x, L, 6).view(-1)).item())
            #     print(certainty)
            # if i != len(self.out_blocks) - 1:
            #     local_lte.append(self.local_lte_fn(x, L, i + 7))
            
            # if i == layer - 8:
            #     return self.output_forward(x, L), inner_state, local_lte, lte, j+1
            # print("layer {}: ".format(i+7), torch.mean(lte_val.view(-1)))
            if not is_train:
                # print(torch.mean(lte_val.view(-1)))
                if torch.mean(lte_val.view(-1)) > 0.9:
                    # print(torch.mean(lte_val.view(-1)))
                    j += i
                    with open("layer.txt", "a") as f:
                        f.write(str(j+1) + "\n")
                    # print("return at out: {}".format(i))
                    return self.output_forward(x, L), inner_state, lte, j+1
        j += i + 1
        with open("layer.txt", "a") as f:
                        f.write(str(13) + "\n")
        x = self.output_forward(x, L)
        # if nsr is not None and cum_alpha is not None:
        #     quality = self.quality_actn(self.quality(x.view(x.shape[0], -1).detach()))
        # print(quality)
        return x, inner_state, local_lte, lte, j
