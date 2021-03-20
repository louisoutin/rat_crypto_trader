import time
import torch
from .helpers import train_one_step, make_std_mask


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index,
              loss_compute, evaluate_loss_compute, is_trn=True, evaluate=True, device="cpu"):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value = 0
    for i in range(total_step):
        if is_trn:
            model.train()
            loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device)
            total_loss += loss.item()
        if i % output_step == 0 and is_trn:
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            start = time.time()
        #########################################################tst########################################################
        tst_total_loss = 0
        with torch.no_grad():
            if i % output_step == 0 and evaluate:
                model.eval()
                tst_loss, tst_portfolio_value = test_batch(DM, x_window_size, model, evaluate_loss_compute,
                                                           local_context_length, device)
                #                tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y)
                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                      (i, tst_loss.item(), tst_portfolio_value.item(), 1 / elapsed))
                start = time.time()

                if tst_portfolio_value > max_tst_portfolio_value:
                    max_tst_portfolio_value = tst_portfolio_value
                    torch.save(model, model_dir + '/' + str(model_index) + ".pkl")
                    #    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
                    print("save model!")
    return tst_loss, tst_portfolio_value


def test_batch(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device):
    tst_batch = DM.get_test_set()
    tst_batch_input = tst_batch["X"]  # (128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float).to(device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)  # [2426, 1, 11]
    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))
    tst_src = torch.tensor(tst_batch_input, dtype=torch.float).to(device)
    tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
    tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
    #############################################################################
    if local_context_length > 1:
        padding_price = tst_currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]  # (11,128,8,4)
    else:
        padding_price = None
    #########################################################################

    tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float).to(device)
    ###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w,  # [128,1,11]   [128, 11, 31, 4])
                            tst_src_mask, tst_trg_mask, padding_price)

    tst_loss, tst_portfolio_value = evaluate_loss_compute(tst_out, tst_trg_y)
    return tst_loss, tst_portfolio_value
