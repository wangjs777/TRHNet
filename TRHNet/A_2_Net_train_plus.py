#import msadapter.pytorch as torch
import torch
#import msadapter.pytorch.nn as nn
import numpy as np
import os
import time
import torch.nn as nn
import utils.evaluate as evaluate
#from mindspore.experimental import optim
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
from models.A_2_net_loss import EndToEnd_A_2_net_Loss
# from data.data_loader import sample_dataloader
from utils import AverageMeter
#import models.A_2_net as A_2_net
import models.A_2_net_plus_NEW as A_2ON_net
import models.A_2_net_plus_NEW188 as A_2ON_net_18
import psutil  # [新增] 用于计算 RAM 占用

# [新增] 辅助函数：获取 RAM 占用
def _get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# [新增] 辅助函数：获取 GPU 占用
def _get_gpu_usage_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def train_e2e(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, att_size, feat_size = args.num_classes, 3, 2048
    model = A_2ON_net_18.A_2_net(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                              device=args.device, pretrained=True)
    model.to(args.device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)
    
    criterion = EndToEnd_A_2_net_Loss(code_length, num_classes=args.num_classes, margin=args.margin)
    criterion.to(args.device)
    cross = nn.CrossEntropyLoss()

    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    
    total_losses, hash_losses, cross_loss = AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()
    best_mAP = 0
    
    # 【核心改变 2】：拆除 max_iter 和 max_epoch 的双层嵌套，改为单层 Epoch 循环
    for epoch in range(args.max_epoch):
        model.train()
        total_losses.reset()
        hash_losses.reset()
        cross_loss.reset()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, (data, targets, index) in pbar:
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            optimizer.zero_grad()
            
            if targets.dim() == 2:
                targets1 = targets.argmax(dim=1)
            else:
                targets1 = targets.long()
                
            # 前向传播
            hash_features, _, _, _, _, cls, cls1, cls2, cls3 = model(data, targets)
            hash_loss = criterion(hash_features, targets1)
            
            # 分类损失 (Classification Loss)
            cls_loss = (1.0 / 2.0) * cross(cls, targets1) + \
                       (1.0 / 6.0) * (cross(cls1, targets1) + cross(cls2, targets1) + cross(cls3, targets1))
            
            # 总损失仅保留代理哈希损失和分类损失
            total_loss = hash_loss + cls_loss
                
            total_loss.backward()
            optimizer.step()
            
            total_losses.update(total_loss.item())
            hash_losses.update(hash_loss.item())
            cross_loss.update(cls_loss.item())
            
            pbar.set_description(f'Epoch [{epoch+1}/{args.max_epoch}] Loss: {total_loss.item():.4f}')
            
        scheduler.step()
        logger.info(
            f'[epoch:{epoch+1}/{args.max_epoch}]'
            f'[total_loss:{total_losses.avg:.4f}]'
            f'[hash_loss:{hash_losses.avg:.4f}]'
            f'[cls_loss:{cross_loss.avg:.6f}]'
        )

        # 【核心改变 4】：不再使用 DCC 更新 B 矩阵。评估直接对网络的输出进行 sign() 量化
        if (epoch + 1) % args.eval_freq == 0:
            test_start = time.time()
            
            # generate_code 函数内部应该包含类似 torch.sign(model(data)) 的操作
            query_code, query_forward_time = generate_code(model, query_dataloader, code_length, args.device)
            database_code, _ = generate_code(model, retrieval_dataloader, code_length, args.device)
            # [新增] 计算 Avg Test Retrieval Time (ms/img)
            avg_test_retrieval_time = (query_forward_time / len(query_dataloader.dataset)) * 1000
            
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                database_code.to(args.device),
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets,
                args.device,
                args.topk,
            )
            test_time = time.time() - test_start
            # [新增] 获取 RAM 和 GPU 占用
            ram_usage = _get_ram_usage_mb()
            gpu_usage = _get_gpu_usage_mb()

            logger.info(f'[epoch:{epoch+1}][code_length:{code_length}][mAP:{mAP:.5f}][test_time:{test_time:.2f}s]')
            # [新增] 打印新增的三个指标
            metrics_line = (
                f"Epoch {epoch+1} | "
                # f"Total Train Time: {total_train_time:.3f} s | " 
                f"Avg Test Retrieval Time: {avg_test_retrieval_time:.4f} ms/img | " 
                f"RAM Usage: {ram_usage:.1f} MB | GPU Usage: {gpu_usage:.1f} MB"
            )
            logger.info(metrics_line)
            
            if mAP > best_mAP:
                best_mAP = mAP
                ret_path = os.path.join('checkpoints_SPBH_18_vegfru', args.info, str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))
                torch.save(database_code.cpu(), os.path.join(ret_path, 'database_code.t'))
                torch.save(query_dataloader.dataset.get_onehot_targets(), os.path.join(ret_path, 'query_targets.t'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.t'))
                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pkl'))
            logger.info('[epoch:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(epoch+1, args.max_epoch, code_length, mAP, best_mAP))    
    logger.info(f'[Training time:{time.time()-start:.2f}]')
    return best_mAP

def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code and calculate total forward propagation time

    Args
        model: The neural network model.
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
        total_forward_time(float): Total time taken for forward propagation (in seconds).
    """
    model.eval()
    total_forward_time = 0.0  # [新增] 初始化前向传播总时间
    
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).to(device)
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            
            # [新增] 记录前向传播开始时间
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.perf_counter()
            
            # 前向传播
            hash_code = model(data, targets)
            
            # [新增] 记录前向传播结束时间
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_t = time.perf_counter()
            
            total_forward_time += (end_t - start_t)  # [新增] 累加该 batch 的耗时
            
            # 💡 提示：如果你在 test/eval 阶段模型返回的是包含多个输出的元组
            # （例如 F_out, dret, all_f...），你需要像下面这样取第一个用于哈希的特征：
            # if isinstance(hash_code, tuple):
            #     hash_code = hash_code[0]
                
            # code[index, :] = hash_code.sign()
            code[index, :] = hash_code
            
    # [修改] 返回哈希码和总时间
    return code, total_forward_time
