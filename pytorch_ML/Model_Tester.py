import torch

class Model_Tester():
    def __init__(self,net,path_to_save_file,device='cuda:0'):
        super().__init__()
        self.net = net
        self.ckpt_dict = torch.load(path_to_save_file)
        self.net.load_state_dict(self.ckpt_dict['state_dict'])
        self.net.to(device)
        self.net.eval()
        self.device = device

    def get_evaluation(self,inputs):
        return self.net(inputs)
