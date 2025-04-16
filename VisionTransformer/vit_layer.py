import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x
    
class ViTEmbedding(nn.Module):
    def __init__(self,img_size, patch_size, in_channels, embed_dim,dropout_rate=0.1):
        super(ViTEmbedding,self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        # CLS 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Leanable Positional Embedding
        # (패치수 + 1) 만큼의 포지셔널 인코딩. 학습가능한 파라미터로 설정
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches +1 , embed_dim))

        # Dropout Layer (Transformer 입력 직전에 적용)
        # 입력 값에 regulaization 적용
        self.dropout = nn.Dropout(dropout_rate)

        # 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x) # [B,N,D]
        B, N, D = x.shape

        # CLS 토큰을 배치 수 만큼 복제해서 맨 앞에 추가
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding 추가
        x = x + self.pos_embedding

        return self.dropout(x) # [B,N+1,D] (N+1 = N + CLS token)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim, num_heads, mlp_ratio=4.0,drop_out=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-head Self Attention
        self.attn = nn.MultiheadAttention(embed_dim,num_heads,dropout=drop_out,batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        # Multi-head Self Attention + residual
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x,x,x) # query, key, value
        x = x_res + attn_output # Residual Connection

        # MLP + residual
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)

        return x