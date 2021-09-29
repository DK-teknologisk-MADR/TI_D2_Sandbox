from detectron2.config import configurable,get_cfg
from detectron2.data.samplers import InferenceSampler
from torch.utils.data import RandomSampler, SequentialSampler
# Usage 2: Decorator on any function. Needs an extra from_config argument:
cfg = None
if cfg is None:
    cfg = get_cfg()
cfg.SOLVER.BASE_LR
@configurable(from_config=lambda cfg: {"a": cfg.VERSION, "b": cfg.SOLVER.BASE_LR})
def a_func(a, b=2, c=3):
    print(a,b,c)
