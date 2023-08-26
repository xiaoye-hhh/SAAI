import torch


def getNewFeature(x, y, k1, k2, mean: bool = False):
    dismat = x @ y.T
    val, rank = dismat.topk(k1)
    dismat[dismat < val[:, -1].unsqueeze(1)] = 0
    if mean:
        dismat = dismat[rank[:, :k2]].mean(dim=1)
    return dismat


def AIM(qf: torch.tensor, gf: torch.tensor, k1, k2):
    qf = qf.to('cuda')
    gf = gf.to('cuda')

    qf = torch.nn.functional.normalize(qf)
    gf = torch.nn.functional.normalize(gf)

    new_qf = torch.concat([getNewFeature(qf, gf, k1, k2)], dim=1)
    new_gf = torch.concat([getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

    new_qf = torch.nn.functional.normalize(new_qf)
    new_gf = torch.nn.functional.normalize(new_gf)

    # additional use of relationships between query sets
    # new_qf = torch.concat([getNewFeature(qf, qf, k1, k2, mean=True), getNewFeature(qf, gf, k1, k2)], dim=1)
    # new_gf = torch.concat([getNewFeature(gf, qf, k1, k2), getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

    return (-new_qf @ new_gf.T - qf @ gf.T).to('cpu')