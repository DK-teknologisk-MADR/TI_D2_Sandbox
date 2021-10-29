from detectron2_ML.hooks import StopFakeExc
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader,build_detection_test_loader
from detectron2_ML.hooks import StopByProgressHook
class TI_Trainer(DefaultTrainer):
    '''
    We should always use this trainer instead of defaulttrainer. Its exactly like it, except that it catches the StopFakeExc used for early stopping by hooks in hooks.py
    '''
    def __init__(self,cfg):
        super().__init__(cfg)

    def train(self,**kwargs):
        try:
            super().train()
        except StopFakeExc:
            self.handle_stop(**kwargs)
        else:
            self.handle_else(**kwargs)

    def handle_stop(self,**kwargs):
        pass

    def handle_else(self,**kwargs):
        pass

class TrainerPeriodicEval(TI_Trainer):
    """
    Trainer with coco-evaluator implemented. if cfg.DATASETS.TEST is filled and
    cfg.TEST.evalperiod>0 then it will evaluate periodically during training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)




class TrainerWithMapper(TI_Trainer):
    '''
    Example of a trainer that applies argumentations at runtime. Argumentations available can be found here:
    https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    '''
    def __init__(self,augmentations,**params_to_DefaultTrainer):
        self.augmentations=augmentations
        super().__init__(**params_to_DefaultTrainer)

    #overwrites default build_train_loader
    def build_train_loader(self, cfg):
          mapper = DatasetMapper(cfg, is_train=True, augmentations=self.augmentations)
          return build_detection_train_loader(cfg,mapper=mapper)




    @classmethod
    def build_evaluator(cls,cfg,dataset_name,output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), False, output_dir=output_folder)



class Trainer_With_Early_Stop(TI_Trainer):
    '''
    Example of a trainer that applies argumentations at runtime. Argumentations available can be found here:
    https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    '''
    def __init__(self,cfg,augmentations = None,patience = None,**params_to_Trainer):
        if augmentations is None:
            self.augmentations = []
        else:
            self.augmentations = augmentations
        print("TRAINER: Augmentations passed to trainer : ",augmentations)
        self.period_between_evals = cfg.TEST.EVAL_PERIOD
        if patience is None:
            self.patience = self.period_between_evals * 15
        else:
            self.patience = patience
        print("StopByProgressHooK: Recieved eval period",self.period_between_evals)
        self.top_score_achieved = 0
        super().__init__(cfg=cfg,**params_to_Trainer)
    #overwrites default build_train_loader

    def build_train_loader(self, cfg):
          mapper = DatasetMapper(cfg, is_train=True, augmentations=self.augmentations)
          return build_detection_train_loader(cfg,mapper=mapper)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = build_detection_test_loader(cfg, dataset_name)
#        for data in iter(loader):
#            print(data)
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls,cfg,dataset_name,output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, ('bbox', 'segm'), False, output_dir=output_folder)


    def build_hooks(self):
        res = super().build_hooks()
        res.append(StopByProgressHook(patience=self.patience, delta_improvement=0.5, score_storage_key='segm/AP', save_name_base="best_model"))
        print("BUILDING THESE HOOKS")
        return res

    def helper_after_train(self,**kwargs):
        self.top_score_achieved = self.storage.latest()[f'best_segm/AP']

    def handle_stop(self,**kwargs):
        self.helper_after_train()

    def handle_else(self,**kwargs):
        self.helper_after_train()


class Hyper_Trainer(Trainer_With_Early_Stop):
    def __init__(self,cfg,augmentations = None,**params_to_Trainer):
        self.patience = 9999999999
        super().__init__(cfg=cfg,augmentations = augmentations,**params_to_Trainer)