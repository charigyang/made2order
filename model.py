import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(f, h, w, dim, device='cuda', temperature = 10000, dtype = torch.float32):
    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class XPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, target, **kwargs):
        return self.fn(self.norm(x), self.norm(target), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, target):
        q = self.to_q(x)#.chunk(3, dim = -1)
        k, v = self.to_kv(target).chunk(2, dim=-1)
        qkv = q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                XPreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, target):
        for xattn, ff in self.layers:
            x = xattn(x,target) + x
            x = ff(x) + x
        return x

class Made2Order(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        decoder_dim,
        dim,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = image_patch_size
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )


        self.pos_embedding = posemb_sincos_3d(frames, image_height // patch_height,image_width // patch_width,dim)
        self.pos_embedding = rearrange(self.pos_embedding, '(f hw) d -> 1 f hw d', f=frames)

        self.T1 = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
        self.T2 = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
        self.T3 = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
        self.T4 = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
        self.T5 = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
        self.T6 = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)

        self.projection_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, decoder_dim)
        )

        self.query = torch.nn.Parameter(to_cuda(torch.randn(1, frames, decoder_dim)))
        self.decoder = TransformerDecoder(decoder_dim, 3, heads, dim_head, mlp_dim, dropout)

    def forward(self, video):
        video = rearrange(video, 'b f c h w -> b c f h w')
        
        #patch+pe
        x = self.to_patch_embedding(video)
        b, f, h, w, d = x.shape
        x = rearrange(x, 'b f h w d -> b f (h w) d') + self.pos_embedding[:, :f]
  
        #Transformer Encoder
        x = rearrange(x, 'b f n d -> (b f) n d')
        x = self.T1(x)
        x = rearrange(x, '(b f) n d -> b f n d', b = b)
        x = rearrange(x, 'b f n d -> (b n) f d')
        x = self.T2(x)
        x = rearrange(x, '(b n) f d -> b f n d', b = b)
        x = rearrange(x, 'b f n d -> (b f) n d')
        x = self.T3(x)
        x = rearrange(x, '(b f) n d -> b f n d', b = b)
        x = rearrange(x, 'b f n d -> (b n) f d')
        x = self.T4(x)
        x = rearrange(x, '(b n) f d -> b f n d', b = b)
        x = rearrange(x, 'b f n d -> (b f) n d')
        x = self.T5(x)
        x = rearrange(x, '(b f) n d -> b f n d', b = b)
        x = rearrange(x, 'b f n d -> (b n) f d')
        x = self.T6(x)
        x = rearrange(x, '(b n) f d -> b f n d', b = b)
        x = rearrange(x, 'b f n d -> b (f n) d', b = b)
        te_output = self.projection_head(x)

        #Transformer Decoder
        query = self.query.repeat(b,1,1)
        td_output = self.decoder(query, te_output)

        #similarity
        te_output = F.normalize(te_output, dim=2)
        td_output = F.normalize(td_output, dim=2)
        output = torch.einsum('bik,bjk->bij', te_output, td_output) 
        attn = rearrange(output, 'b (f n) d -> b f n d', f=f)

        output = torch.max(attn, 2)[0]
        return output, attn
