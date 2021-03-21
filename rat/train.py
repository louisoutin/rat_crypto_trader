import time
import torch
from .helpers import make_std_mask


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index,
              loss_func, eval_loss_func, optimizer, device="cpu"):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    # TRAINING
    max_tst_portfolio_value = 0
    for i in range(total_step):
        model.train()
        out, trg_y = train_one_step(DM, x_window_size, model, local_context_length, device)
        loss, portfolio_value = loss_func(out, trg_y)
        loss.backward()
        optimizer.step()
        optimizer.optimizer.zero_grad()
        total_loss += loss.item()
        if i % output_step == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            start = time.time()

        # VALIDATION
        tst_total_loss = 0
        if i % output_step == 0:
            with torch.no_grad():
                model.eval()
                tst_out, tst_trg_y = test_batch(DM, x_window_size, model, local_context_length, device)
                tst_loss, tst_portfolio_value = eval_loss_func(tst_out, tst_trg_y)
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


def train_one_step(DM, x_window_size, model, local_context_length, device):
    batch = DM.next_batch()
    batch_input = batch["X"]  # (128, 4, 11, 31)
    batch_y = batch["y"]  # (128, 4, 11)
    batch_last_w = batch["last_w"]  # (128, 11)
    batch_w = batch["setw"]
    #############################################################################
    previous_w = torch.tensor(batch_last_w, dtype=torch.float, device=device)
    previous_w = torch.unsqueeze(previous_w, 1)  # [128, 11] -> [128,1,11]
    batch_input = batch_input.transpose((1, 0, 2, 3))
    batch_input = batch_input.transpose((0, 1, 3, 2))
    src = torch.tensor(batch_input, dtype=torch.float, device=device)
    price_series_mask = (torch.ones(src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
    currt_price = src.permute((3, 1, 2, 0))  # [4,128,31,11]->[11,128,31,4]
    if local_context_length > 1:
        padding_price = currt_price[:, :, -local_context_length * 2 + 1:-1, :]
    else:
        padding_price = None
    currt_price = currt_price[:, :, -1:, :]  # [11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price, src.size()[1])
    batch_y = batch_y.transpose((0, 2, 1))  # [128, 4, 11] ->#[128,11,4]
    trg_y = torch.tensor(batch_y, dtype=torch.float, device=device)
    out = model.forward(src, currt_price, previous_w,
                        price_series_mask, trg_mask, padding_price)
    new_w = out[:, :, 1:]  # 去掉cash
    new_w = new_w[:, 0, :]  # #[109,1,11]->#[109,11]
    new_w = new_w.detach().cpu().numpy()
    batch_w(new_w)

    return out, trg_y


def test_batch(DM, x_window_size, model, local_context_length, device):
    tst_batch = DM.get_test_set()
    tst_batch_input = tst_batch["X"]  # (128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float, device=device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)  # [2426, 1, 11]
    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))
    tst_src = torch.tensor(tst_batch_input, dtype=torch.float, device=device)
    tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
    tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
    #############################################################################
    if local_context_length > 1:
        padding_price = tst_currt_price[:, :, -local_context_length * 2 + 1:-1, :]  # (11,128,8,4)
    else:
        padding_price = None
    #########################################################################

    tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float, device=device)
    ###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w,  # [128,1,11]   [128, 11, 31, 4])
                            tst_src_mask, tst_trg_mask, padding_price)
    return tst_out, tst_trg_y
