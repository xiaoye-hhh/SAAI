import torch
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = dist_an.data > dist_ap.data
        length = torch.sqrt((inputs * inputs).sum(1)).mean()
        return loss, dist_ap, dist_an

class TripletHardLoss(nn.Module):
    def __init__(self, margin=0.3) -> None:
        """ 难三元组损失: N*K个样本(N是不同pid; k是每个pid的不同图片)
                找出将每个样本当作anchor，寻找和anchor距离最大的同pid的图片，
                以及和anchor距离最小的不同pid的图片，计算loss

        Args:
            margin (float, optional): 希望拉开的距离. Defaults to 0.3.
        """
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: torch.Tensor, label: torch.Tensor):
        N = features.shape[0]

        dist = features.pow(2).sum(dim=1, keepdim=True).expand(N, N) + features.pow(2).sum(dim=1, keepdim=True).expand(N, N).T
        dist = dist - 2 * features @ features.T
        dist = dist.clamp(min=1e-12).sqrt()

        same_id_mask = label.expand(N, N).eq(label.expand(N, N).T)
        dp, _ = dist[same_id_mask].view(N, -1).max(dim=1)
        dn, _ = dist[~same_id_mask].view(N, -1).min(dim=1)
        target = torch.ones_like(dp)

        return self.loss_fn(dn, dp, target), None, None