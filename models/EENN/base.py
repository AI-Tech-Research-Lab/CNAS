from abc import ABC, abstractmethod
import torch
from torch import nn


class IntermediateBranch(nn.Module):
    def __init__(self, classifier: nn.Module,
                 preprocessing: nn.Module = None):
        super().__init__()
        self.preprocessing = preprocessing if preprocessing is not None \
            else lambda x: x

        self.classifier = classifier

    def preprocess(self, x):
        return self.preprocessing(x)

    def logits(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        return logits

    def forward(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        return logits


class BinaryIntermediateBranch(IntermediateBranch):
    def __init__(self, classifier: nn.Module,
                 # binary_classifier: nn.Module = None,
                 # constant_binary_output=None,
                 preprocessing: nn.Module = None,
                 return_one=False):

        super().__init__(classifier, preprocessing)

        # self.c1 = deepcopy(self.classifier)
        # self.c1.add_module('bin', nn.Linear(10, 1))
        # self.c1.add_module('s', nn.Sigmoid())
        self.return_one = return_one

        # if binary_classifier is None and constant_binary_output is None:
        #     assert False

        # if binary_classifier is None:
        #     if not isinstance(constant_binary_output, (float, int)):
        #         assert False
        #
        #     binary_classifier = lambda x: torch.full((x.shape[0], 1),
        #                                              constant_binary_output,
        #                                              device=x.device)
        #
        # self.binary_classifier = binary_classifier

    def logits(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        if not self.return_one:
            logits = logits[:, :-1]

        return logits

    def forward(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        if not self.return_one:
            logits, bin = logits[:, :-1], logits[:, -1:]
            bin = torch.sigmoid(bin)
            # bin = torch.minimum(bin, torch.ones_like(bin))
            # bin = torch.maximum(bin, torch.zeros_like(bin))
            # if not self.training:
            #     bin = 1 - bin
        else:
            bin = torch.ones((x.shape[0], 1), device=x.device)

        # if self.return_one:
        #     bin = torch.ones_like(bin)
        # bin = self.c1(embs)
        # else:
        # bin = self.binary_classifier(logits)

        return logits, bin


class BranchModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def n_branches(self):
        raise NotImplemented


