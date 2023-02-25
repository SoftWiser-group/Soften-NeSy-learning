import sys 
sys.path.append("..") 
from nn_utils import *
from .sym_net import SymbolNet

class NNAOG(nn.Module):
    def __init__(self):
        super(NNAOG, self).__init__()
        self.sym_net = SymbolNet()

    
    def forward(self, img_seq, mask=True):
        # mask indicates whether to mask incorrect grammar
        batch_size = img_seq.shape[0]
        max_len = img_seq.shape[1]
        images = img_seq.reshape((-1, img_seq.shape[-3], img_seq.shape[-2], img_seq.shape[-1]))
        logits = self.sym_net(images)
        logits = logits.reshape((batch_size, max_len, -1))

        if mask:
            mask = torch.zeros_like(logits, device=logits.device)
            digit_pos_list = np.arange(0, max_len, 2)
            op_pos_list = np.arange(1, max_len, 2)
            mask[:, digit_pos_list[:, None], digit_idx_list] = 1.
            if len(op_pos_list) > 0:
                mask[:, op_pos_list[:, None], op_idx_list] = 1. 
            masked_logits = mask * logits
            masked_logits[(1-mask).bool()] += -1e10
            return masked_logits
        else:
            return logits.reshape((batch_size, max_len, -1))


