import torch
from torch import nn
import train_utils.distributed_utils as utils
import numpy as np
from PIL import Image

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=8)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    color_map = {0: (0, 0, 0), 1: (0, 0, 255), 2:(0, 255, 0),
                          3: (0, 255, 255), 4: (255, 0, 0), 5 :(255, 0, 255),
                          6: (255, 255, 0), 7: (255, 255, 255)}

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        cnt = 0
        # for i in range(16):
        #     img = image[i].detach().cpu().numpy().transpose(1,2,0)
        #     img = Image.fromarray((img * 255).astype(np.uint8))
        #     img.save("/home/user/mix/seg_train/mi_im/" + str(cnt) + "img.jpg")

        #     tgt = target[i].detach().cpu().numpy()
        #     res_tgt = np.zeros_like(tgt)
        #     res_tgt = np.stack((res_tgt, res_tgt, res_tgt), axis=2)
        #     for i in range(tgt.shape[0]):
        #         for j in range(tgt.shape[1]):
        #             res_tgt[i][j] = np.asarray(color_map[tgt[i][j]])

        #     res_tgt = Image.fromarray(res_tgt.astype(np.uint8))
        #     res_tgt.save("/home/user/mix/seg_train/mi_im/" + str(cnt) + "target.jpg")
        #     cnt += 1
        #     print(cnt)
        # continue
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
    # return 1, 0
    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
