"""CCG pair auto-classification: dataset, models, training, and verification."""
from neuropy.classifier.dataset import LabeledSet, PairSample, build_labeled_set
from neuropy.classifier.models import MODELS, UNSURE, BaseModel, decide
from neuropy.classifier.train import fit_final, leave_one_rat_out, report

__all__ = ['LabeledSet', 'PairSample', 'build_labeled_set', 'MODELS', 'UNSURE',
           'BaseModel', 'decide', 'fit_final', 'leave_one_rat_out', 'report']
