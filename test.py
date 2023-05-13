import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F

def test(model, data, train_mask, val_mask, test_mask, alpha):
    with torch.no_grad():
        model.eval()
        logits, accs = model(data, alpha), []
        for mask in [train_mask,val_mask,test_mask]:
            pred = logits[mask].max(1)[1]
            acc = f1_score(pred.cpu(), data.y[mask].cpu(), average='micro')
            accs.append(acc)

        accs.append(F.nll_loss(model(data, alpha)[val_mask], data.y[val_mask]))
        accs.append(f1_score(pred.cpu(), data.y[mask].cpu(), average='weighted'))
    return accs