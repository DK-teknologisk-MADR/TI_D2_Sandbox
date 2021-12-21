import torch.nn as nn

class OSC_Loss():
    def __init__(self,nr_of_classes):
        self.nr_of_classes = nr_of_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.CNloss = nn.CrossEntropyLoss(ignore_index=self.nr_of_classes ,reduction='none')
        self.nr_of_classes = nr_of_classes

    def __call__(self,outs,targets):
        assert targets.ndim == 2 #ys must be unsqueezed so that it is B x 1
        assert outs.ndim == 2
        cross_entropy_loss = self.CNloss(outs[:,:self.nr_of_classes],targets.flatten().long())
        mean_soft_max_loss = -self.logsoftmax(outs[:,:self.nr_of_classes]).mean(axis=1)
        loss = (targets < self.nr_of_classes).flatten() * cross_entropy_loss + (targets == self.nr_of_classes).flatten() * mean_soft_max_loss
        return loss.mean()
