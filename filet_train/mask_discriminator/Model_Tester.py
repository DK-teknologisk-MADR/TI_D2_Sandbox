import torch

class Model_Tester(torch.nn.Module):
    def __init__(self,net,path_to_save_file,device='cuda:0'):
        super().__init__()
        self.net = net
        self.net.eval()
        ckpt_dict = torch.load(path_to_save_file)
        self.net.load_state_dict(ckpt_dict['state_dict'])
        self.net.to(device)
        self.device = device

    def get_evaluation(self):
        raise NotImplementedError

