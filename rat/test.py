import time
import torch
from .helpers import train_one_step, make_std_mask


def test_online(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device):
    tst_batch = DM.get_test_set_online(DM._test_ind[0], DM._test_ind[-1], x_window_size)
    tst_batch_input = tst_batch["X"]
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float).to(device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    long_term_tst_src = torch.tensor(tst_batch_input, dtype=torch.float).to(device)
    #########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1], 1, x_window_size) == 1)

    long_term_tst_currt_price = long_term_tst_src.permute((3, 1, 2, 0))
    long_term_tst_currt_price = long_term_tst_currt_price[:, :, x_window_size - 1:, :]
    ###############################################################################################
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:, :, 0:1, :], long_term_tst_src.size()[1])

    tst_batch_y = tst_batch_y.transpose((0, 3, 2, 1))
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float).to(device)
    tst_long_term_w = []
    tst_y_window_size = len(DM._test_ind) - x_window_size - 1 - 1
    for j in range(tst_y_window_size + 1):  # 0-9
        tst_src = long_term_tst_src[:, :, j:j + x_window_size, :]
        tst_currt_price = long_term_tst_currt_price[:, :, j:j + 1, :]
        if (local_context_length > 1):
            padding_price = long_term_tst_src[:, :, j + x_window_size - 1 - local_context_length * 2 + 2:j + x_window_size - 1, :]
            padding_price = padding_price.permute((3, 1, 2, 0))  # [4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price = None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w,
                            # [109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                            tst_src_mask, tst_trg_mask, padding_price)
        if (j == 0):
            tst_long_term_w = out.unsqueeze(0)  # [1,109,1,12]
        else:
            tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
        out = out[:, :, 1:]  # 去掉cash #[109,1,11]
        tst_previous_w = out
    tst_long_term_w = tst_long_term_w.permute(1, 0, 2, 3)  ##[10,128,1,12]->#[128,10,1,12]
    tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = evaluate_loss_compute(tst_long_term_w, tst_trg_y)
    return tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO


def test_net(DM, total_step, output_step, x_window_size, local_context_length, model, loss_compute,
             evaluate_loss_compute, is_trn=True, evaluate=True, device="cpu"):
    "Standard Testing and Logging Function"
    start = time.time()
    total_loss = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value = 0

    for i in range(total_step):
        if is_trn:
            loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            start = time.time()
        #########################################################tst########################################################
        tst_total_loss = 0
        with torch.no_grad():
            if (i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = test_online(DM, x_window_size, model,
                                                                                            evaluate_loss_compute,
                                                                                            local_context_length,
                                                                                            device)
                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                      (i, tst_loss.item(), tst_portfolio_value.item(), SR.item(), CR.item(), TO.item(), 1 / elapsed))
                start = time.time()
                #                portfolio_value_list.append(portfolio_value.item())

                if (tst_portfolio_value > max_tst_portfolio_value):
                    max_tst_portfolio_value = tst_portfolio_value
                    log_SR = SR
                    log_CR = CR
                    log_St_v = St_v
                    log_tst_pc_array = tst_pc_array
    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array, TO
