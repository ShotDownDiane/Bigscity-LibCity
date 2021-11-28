import torch.nn as nn
import torch

class Attn(nn.Module):
    def __init__(self, max_len, emb_loc, loc_max, device, dropout=0.1):
        super(Attn, self).__init__()
        self.max_len = max_len
        self.value = nn.Linear(self.max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max
        self.device = device

    def forward(self, self_attn, self_delta, traj_len):
        # self_attn (N, M, emb), candidate (N, L, emb), \
        # self_delta (N, M, L, emb), len [N]
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)
        # squeeze the embed dimension
        [N, L, M] = self_delta.shape
        candidates = torch.linspace(
            1, int(self.loc_max), int(self.loc_max)
        ).long()  # (L)
        candidates = candidates.unsqueeze(0).expand(N, -1).to(self.device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.mul(
            torch.bmm(emb_candidates, self_attn.transpose(-1, -2)),
            self_delta)  # (N, L, M)
        # pdb.set_trace()
        attn_out = self.value(attn).view(N, L)  # (N, L)
        # attn_out = F.log_softmax(attn_out, dim=-1)
        # ignore if cross_entropy_loss

        return attn_out  # (N, L)