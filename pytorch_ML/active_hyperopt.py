from hyperopt import Hyperopt
class Active_Hyperopt(Hyperopt):
    '''
    NOT finished as of yet. Purpose is to get a leaner that query for now samples after each prune of e.g. SHA

    '''
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def after_prune(self):
        trial_ids_alive = [i for i in range(self.pruner.participants) if not self.result_df['pruned']]
        for trial_id in trial_ids_alive:
            net = self.build_net(self.hyper_vals[trial_id]) #TODO::create this
            self.validate(net) #TODO::create this

