import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,embed_dim,atten_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim,atten_dim,bias=False)
        self.key = nn.Linear(embed_dim,atten_dim,bias=False)
        self.value = nn.Linear(embed_dim,atten_dim,bias=False)

    def forward(self,x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query,key.transpose(-2,-1))
        scores = scores / key.size(-1)**0.5

        attention_weight = F.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weight,value)
        return weighted_values
    
class MultiheadAttention(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()
        attention_dim = embed_dim // num_heads
        self.attentions = nn.ModuleList([SelfAttention(embed_dim,attention_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(embed_dim,embed_dim)

    def forward(self,x):
        head_outputs = []
        for attention in self.attentions:
            head_output = attention(x)
            head_outputs.append(head_output)
        concatenated_heads = torch.cat(head_outputs, dim=-1)
        print("concatenated_heads", concatenated_heads.shape)
        output = self.fc(concatenated_heads)
        print("output", output.shape)
        return output
