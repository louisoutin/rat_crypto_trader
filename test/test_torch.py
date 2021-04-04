import torch


def test_gpu():
    print("GPU?", torch.cuda.is_available())
    t = torch.randn(10, 10).cuda()
    print("tensor", t)
