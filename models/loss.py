import utils

def cross_entropy_loss(scores, targets, criterion, config):
    loss = criterion(scores, targets.view(-1))
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(utils.PAD).data).sum()
    num_total = targets.ne(utils.PAD).data.sum()
    loss = loss / num_total

    return loss, num_total, num_correct
