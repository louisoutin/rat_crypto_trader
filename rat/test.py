import time
import torch
from .helpers import make_std_mask


def test_net(DM, x_window_size, local_context_length, model,
             evaluate_loss_compute, device="cpu"):
    "Standard Testing and Logging Function"
    start = time.time()

    #########################################################tst########################################################
    tst_total_loss = 0
    with torch.no_grad():
        model.eval()
        tst_long_term_w, tst_trg_y = test_online(DM, x_window_size, model,
                                                 local_context_length,
                                                 device)
        tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = evaluate_loss_compute(tst_long_term_w,
                                                                                              tst_trg_y)
        tst_total_loss += tst_loss.item()
        elapsed = time.time() - start
        print("Test Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
              (tst_loss.item(), tst_portfolio_value.item(), SR.item(), CR.item(), TO.item(), 1 / elapsed))

    return tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO


def test_online(DM, x_window_size, model, local_context_length, device):
    tst_batch = DM.get_test_set_online(DM.indices[0], DM.indices[-1], x_window_size)
    tst_batch_input = tst_batch["X"]
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float, device=device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    long_term_tst_src = torch.tensor(tst_batch_input, dtype=torch.float, device=device)
    #########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1], 1, x_window_size) == 1)

    long_term_tst_currt_price = long_term_tst_src.permute((3, 1, 2, 0))
    long_term_tst_currt_price = long_term_tst_currt_price[:, :, x_window_size - 1:, :]
    ###############################################################################################
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:, :, 0:1, :], long_term_tst_src.size()[1])

    tst_batch_y = tst_batch_y.transpose((0, 3, 2, 1))
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float, device=device)
    tst_long_term_w = []
    tst_y_window_size = len(DM.indices) - x_window_size - 1 - 1
    for j in range(tst_y_window_size + 1):  # 0-9
        tst_src = long_term_tst_src[:, :, j:j + x_window_size, :]
        tst_currt_price = long_term_tst_currt_price[:, :, j:j + 1, :]
        if local_context_length > 1:
            padding_price = long_term_tst_src[:, :, j + x_window_size - 1 - local_context_length * 2 + 2:j + x_window_size - 1, :]
            padding_price = padding_price.permute((3, 1, 2, 0))  # [4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price = None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w,
                            # [109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                            tst_src_mask, tst_trg_mask, padding_price)
        if j == 0:
            tst_long_term_w = out.unsqueeze(0)  # [1,109,1,12]
        else:
            tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
        out = out[:, :, 1:]  # 去掉cash #[109,1,11]
        tst_previous_w = out
    tst_long_term_w = tst_long_term_w.permute(1, 0, 2, 3)  ##[10,128,1,12]->#[128,10,1,12]
    return tst_long_term_w, tst_trg_y
