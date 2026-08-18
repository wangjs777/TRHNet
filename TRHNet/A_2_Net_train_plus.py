from fvcore.nn import FlopCountAnalysis
import torch
import numpy as np
import os
import time
import torch.nn as nn
import utils.evaluate as evaluate
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
from models.A_2_net_loss import EndToEnd_A_2_net_Loss
from utils import AverageMeter
import models.A_2_net_plus_NEW as A_2ON_net
import models.A_2_net_plus_NEW188 as A_2ON_net_18


def _extract_hash_code(model_output):
    """
    从模型输出中提取哈希特征。

    兼容：
    1. eval阶段直接返回 hash_code
    2. eval阶段仍然返回 tuple/list
    """
    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    if isinstance(model_output, dict):
        for key in ['hash_code', 'hash_codes', 'hash_feature',
                    'hash_features', 'code', 'output']:
            if key in model_output:
                return model_output[key]

        raise RuntimeError(
            "Model returned a dict, but no hash output was found."
        )

    return model_output


def count_inference_params(model, sample_data, sample_targets):
    """
    Params (M):
    统计实际推理前向路径中被调用模块所使用的参数。

    """

    was_training = model.training
    model.eval()

    used_params = {}
    hooks = []

    def forward_hook(module, inputs, output):
        # recurse=False：
        # 只统计当前模块直接拥有的参数，
        # 防止父模块/子模块重复统计
        for param in module.parameters(recurse=False):
            used_params[id(param)] = param.numel()

    # 给所有模块注册hook，记录哪些模块真正执行过
    for module in model.modules():
        hooks.append(
            module.register_forward_hook(forward_hook)
        )

    with torch.no_grad():
        _ = model(sample_data, sample_targets)

    # 删除hook
    for hook in hooks:
        hook.remove()

    inference_params = sum(used_params.values())

    # 完整模型参数，仅用于检查排除了多少训练专用参数
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    if was_training:
        model.train()

    return (
        inference_params / 1e6,
        total_params / 1e6,
        (total_params - inference_params) / 1e6
    )


def calculate_inference_flops(
        model,
        sample_data,
        sample_targets
):
    """
    FLOPs (G):
    单张图片一次真实 inference forward 的理论计算量。

    使用 PyTorch eager FlopCounterMode，
    不使用 fvcore JIT tracing。
    """

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        raise ImportError(
            "Current PyTorch version does not provide "
            "torch.utils.flop_counter.FlopCounterMode. "
            "Please print torch.__version__ first."
        )

    model.eval()

    # 保证输入与模型严格处于相同GPU
    model_device = next(model.parameters()).device

    sample_data = sample_data.to(model_device)
    sample_targets = sample_targets.to(model_device)

    if sample_data.shape[0] != 1:
        raise ValueError(
            f"FLOPs measurement requires batch size 1, "
            f"but got {sample_data.shape[0]}."
        )

    if sample_data.is_cuda:
        torch.cuda.synchronize()

    with torch.no_grad():

        with FlopCounterMode(display=False) as flop_counter:
            # 正常GPU eager forward
            model_output = model(
                sample_data,
                sample_targets
            )

            # FLOPs只统计模型forward
            # sign量化通常不纳入理论模型FLOPs

    if sample_data.is_cuda:
        torch.cuda.synchronize()

    total_flops = flop_counter.get_total_flops()

    flops_g = total_flops / 1e9

    return flops_g, {}


def measure_inference_latency(
        model,
        sample_data,
        sample_targets,
        warmup_runs=50,
        measurement_runs=200
):
    """
    Latency (ms/image):

    测量协议：
    - batch size = 1
    - model.eval()
    - torch.no_grad()
    - 输入Tensor提前放在GPU
    - 不包含DataLoader
    - 不包含CPU -> GPU传输
    - 包含模型forward
    - 包含 sign() 哈希量化
    """

    if sample_data.shape[0] != 1:
        raise ValueError(
            "Latency measurement requires batch_size = 1."
        )

    was_training = model.training
    model.eval()

    with torch.no_grad():

        # ==========================
        # Warm-up
        # ==========================
        for _ in range(warmup_runs):
            model_output = model(
                sample_data,
                sample_targets
            )

            hash_code = _extract_hash_code(
                model_output
            )

            # 与你的generate_code保持完全一致
            _ = hash_code.sign()

        if sample_data.is_cuda:
            torch.cuda.synchronize()

        # ==========================
        # 正式计时
        # ==========================
        start_time = time.perf_counter()

        for _ in range(measurement_runs):
            model_output = model(
                sample_data,
                sample_targets
            )

            hash_code = _extract_hash_code(
                model_output
            )

            # 生成最终哈希码
            binary_code = hash_code.sign()

        if sample_data.is_cuda:
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time

    latency_ms = (
            elapsed_time
            * 1000.0
            / measurement_runs
    )

    if was_training:
        model.train()

    return latency_ms


def measure_model_complexity(
        model,
        dataloader,
        device,
        warmup_runs=50,
        measurement_runs=200
):
    """
    一次性统计：

    1. Params (M)
    2. FLOPs (G)
    3. Latency (ms/image)
    """

    # 从真实测试集获取一个batch
    data, targets, _ = next(iter(dataloader))

    # 强制只取一张图片
    sample_data = data[:1].to(
        device,
        non_blocking=True
    )

    sample_targets = targets[:1].to(
        device,
        non_blocking=True
    )

    # ==============================
    # Params
    # ==============================
    params_m, full_params_m, excluded_params_m = \
        count_inference_params(
            model,
            sample_data,
            sample_targets
        )

    # ==============================
    # FLOPs
    # ==============================
    flops_g, unsupported_ops = \
        calculate_inference_flops(
            model,
            sample_data,
            sample_targets
        )

    # ==============================
    # Latency
    # ==============================
    latency_ms = measure_inference_latency(
        model,
        sample_data,
        sample_targets,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs
    )

    if sample_data.is_cuda:
        device_name = torch.cuda.get_device_name(
            sample_data.device
        )
    else:
        device_name = 'CPU'

    return {
        'params_m': params_m,
        'full_params_m': full_params_m,
        'excluded_params_m': excluded_params_m,
        'flops_g': flops_g,
        'latency_ms': latency_ms,
        'unsupported_ops': unsupported_ops,
        'device_name': device_name,
        'input_shape': tuple(sample_data.shape),
        'warmup_runs': warmup_runs,
        'measurement_runs': measurement_runs
    }


def train_e2e(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, att_size, feat_size = args.num_classes, 3, 2048
    model = A_2ON_net.A_2_net(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                              device=args.device, pretrained=True)
    model.to(args.device)
    # ============================================================
    # Inference Complexity
    # Params / FLOPs / Latency
    # ============================================================

    complexity = measure_model_complexity(
        model=model,
        dataloader=query_dataloader,
        device=args.device,
        warmup_runs=50,
        measurement_runs=200
    )

    logger.info(
        f"[Inference Complexity]"
        f"[code_length:{code_length}]"
        f"[Params:{complexity['params_m']:.3f} M]"
        f"[FLOPs:{complexity['flops_g']:.3f} G]"
        f"[Latency:{complexity['latency_ms']:.4f} ms/image]"
    )

    logger.info(
        f"[Complexity Protocol]"
        f"[device:{complexity['device_name']}]"
        f"[input_shape:{complexity['input_shape']}]"
        f"[batch_size:1]"
        f"[warmup:{complexity['warmup_runs']}]"
        f"[runs:{complexity['measurement_runs']}]"
        f"[data_transfer:excluded]"
        f"[hash_sign:included_in_latency]"
    )

    logger.info(
        f"[Parameter Check]"
        f"[Full Model Params:{complexity['full_params_m']:.3f} M]"
        f"[Inference Params:{complexity['params_m']:.3f} M]"
        f"[Training-only Params:"
        f"{complexity['excluded_params_m']:.3f} M]"
    )

    if complexity['unsupported_ops']:
        logger.warning(
            "Some operators are not counted by fvcore: "
            f"{complexity['unsupported_ops']}"
        )

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen,
                              nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)

    criterion = EndToEnd_A_2_net_Loss(code_length, args.batch_size, num_classes=args.num_classes, margin=args.margin,
                                      finetune=False)
    criterion.to(args.device)
    cross = nn.CrossEntropyLoss()
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    cnn_losses, hash_losses, cross_loss, reconstruction_losses, decorrelation_losses, gan_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()
    best_mAP = 0
    # 【核心改变 2】：拆除 max_iter 和 max_epoch 的双层嵌套，改为单层 Epoch 循环
    for epoch in range(args.max_epoch):
        model.train()
        cnn_losses.reset()
        hash_losses.reset()
        reconstruction_losses.reset()
        decorrelation_losses.reset()
        gan_losses.reset()
        cross_loss.reset()
        if epoch == (args.max_epoch // 2):
            criterion.finetune = True
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, (data, targets, index) in pbar:
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            optimizer.zero_grad()
            if targets.dim() == 2 and targets.size(1) == args.num_classes:
                targets1 = targets.argmax(dim=1)
            else:
                targets1 = targets
            # 前向传播
            F, dret, all_f, deep_S, inputs, cls, cls1, cls2, cls3 = model(data, targets)
            # F, dret, all_f, deep_S, inputs, cls, cls1 = model(data, targets)

            inputs['img'] = data

            cnn_loss, hash_loss, reconstruction_loss, decorrelation_loss, gan_loss = criterion(F, targets1, dret, all_f,
                                                                                               inputs)

            # 1. 计算分类损失 (Classification Loss)
            cls_loss = (1.0 / 2.0) * cross(cls, targets1) + \
                       (1.0 / 6.0) * (cross(cls1, targets1) + cross(cls2, targets1) + cross(cls3, targets1))
            # cls_loss = cross(cls, targets1)

            # 引入类似第一个模型的 Warm-up 策略

            warm_up_epochs = 5

            if epoch < warm_up_epochs:

                # 热身阶段：扣除 hash_loss，但保留分类、重构、去相关和GAN损失

                total_loss = (cnn_loss - hash_loss) + cls_loss

            else:

                # 正常阶段：包含所有损失（你可以调整 gamma 控制 hash_loss 的权重，如果不用直接相加即可）

                total_loss = cnn_loss + cls_loss

            total_loss.backward()

            optimizer.step()

            cnn_losses.update(total_loss.item())

            hash_losses.update(hash_loss.item() if epoch >= warm_up_epochs else 0)

            cross_loss.update(cls_loss.item())

            reconstruction_losses.update(reconstruction_loss.item())

            decorrelation_losses.update(decorrelation_loss.item())

            gan_losses.update(gan_loss.item())

            pbar.set_description(f'Epoch [{epoch + 1}/{args.max_epoch}] Loss: {total_loss.item():.4f}')

        scheduler.step()

        logger.info(

            f'[epoch:{epoch + 1}/{args.max_epoch}]'

            f'[total_loss:{cnn_losses.avg:.4f}]'

            f'[hash_loss:{hash_losses.avg:.4f}]'

            f'[cls_loss:{cross_loss.avg:.6f}]'

            f'[recon_loss:{reconstruction_losses.avg:.4f}]'

            f'[decorr_loss:{decorrelation_losses.avg:.4f}]'

            f'[gan_loss:{gan_losses.avg:.4f}]'

        )

        # 不再使用 DCC 更新 B 矩阵。评估直接对网络的输出进行 sign() 量化

        if (epoch + 1) % args.eval_freq == 0:

            test_start = time.time()

            # generate_code 函数内部应该包含类似 torch.sign(model(data)) 的操作

            query_code = generate_code(model, query_dataloader, code_length, args.device)

            database_code = generate_code(model, retrieval_dataloader, code_length, args.device)

            mAP = evaluate.mean_average_precision(

                query_code.to(args.device),

                database_code.to(args.device),

                query_dataloader.dataset.get_onehot_targets().to(args.device),

                retrieval_targets,

                args.device,

                args.topk,

            )

            test_time = time.time() - test_start

            logger.info(f'[epoch:{epoch + 1}][code_length:{code_length}][mAP:{mAP:.5f}][test_time:{test_time:.2f}s]')

            if mAP > best_mAP:

                best_mAP = mAP

                ret_path = os.path.join('checkpoints_food101', args.info, str(code_length))

                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)

                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))

                torch.save(database_code.cpu(), os.path.join(ret_path, 'database_code.t'))

                torch.save(query_dataloader.dataset.get_onehot_targets(), os.path.join(ret_path, 'query_targets.t'))

                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.t'))

                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pkl'))

            logger.info('[epoch:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(epoch + 1, args.max_epoch,
                                                                                            code_length, mAP, best_mAP))

    logger.info(f'[Training time:{time.time() - start:.2f}]')

    return best_mAP


# def generate_code(model, dataloader, code_length, device):
#     """
#     Generate hash code and calculate total forward propagation time

#     Args
#         model: The neural network model.
#         dataloader(torch.utils.data.DataLoader): Data loader.
#         code_length(int): Hash code length.
#         device(torch.device): Using gpu or cpu.

#     Returns
#         code(torch.Tensor): Hash code.
#         total_forward_time(float): Total time taken for forward propagation (in seconds).
#     """
#     model.eval()
#     with torch.no_grad():
#         N = len(dataloader.dataset)
#         code = torch.zeros([N, code_length]).to(device)
#         for batch, (data, targets, index) in enumerate(dataloader):
#             data, targets, index = data.to(device), targets.to(device), index.to(device)


#             # 前向传播
#             hash_code = model(data, targets)


#             code[index, :] = hash_code.sign()
#             # code[index, :] = hash_code

#     # [修改] 返回哈希码和总时间
#     return code
def generate_code(
        model,
        dataloader,
        code_length,
        device
):
    """
    Generate binary hash codes.
    """

    model.eval()

    with torch.no_grad():
        N = len(dataloader.dataset)

        code = torch.zeros(
            [N, code_length],
            device=device
        )

        for data, targets, index in dataloader:
            data = data.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            index = index.to(device)

            # ======================
            # Forward
            # ======================
            model_output = model(
                data,
                targets
            )

            hash_code = _extract_hash_code(
                model_output
            )

            # ======================
            # Binary hash code
            # ======================
            binary_code = hash_code.sign()

            code[index, :] = binary_code

    return code


def test_complexity(
        query_dataloader,
        code_length,
        args,
        checkpoint_path
):
    """
    Test inference complexity using a trained checkpoint.
    """

    num_classes = args.num_classes
    att_size = 3
    feat_size = 2048

    # =====================================================
    # 1. Build model
    # pretrained=False:
    # because we will immediately load the trained checkpoint
    # =====================================================
    model = A_2ON_net.A_2_net(
        code_length=code_length,
        num_classes=num_classes,
        att_size=att_size,
        feat_size=feat_size,
        device=args.device,
        pretrained=False
    )

    model = model.to(args.device)

    # =====================================================
    # 2. Load trained checkpoint
    # =====================================================
    logger.info(
        f"Loading checkpoint: {checkpoint_path}"
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=args.device
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.eval()

    # =====================================================
    # 3. Measure complexity
    #
    # Paper protocol:
    # warm-up = 100
    # forward passes = 1000
    # batch size = 1
    # =====================================================
    complexity = measure_model_complexity(
        model=model,
        dataloader=query_dataloader,
        device=args.device,
        warmup_runs=100,
        measurement_runs=1000
    )

    logger.info("=" * 80)

    logger.info(
        f"[Inference Complexity]"
        f"[code_length:{code_length}]"
        f"[Params:{complexity['params_m']:.4f} M]"
        f"[FLOPs:{complexity['flops_g']:.3f} G]"
        f"[Latency:{complexity['latency_ms']:.4f} ms/image]"
    )

    logger.info(
        f"[Complexity Protocol]"
        f"[GPU:{complexity['device_name']}]"
        f"[input:{complexity['input_shape']}]"
        f"[batch_size:1]"
        f"[warmup:100]"
        f"[runs:1000]"
        f"[data_loading:excluded]"
        f"[preprocessing:excluded]"
        f"[host_to_device:excluded]"
        f"[sign_quantization:included]"
    )

    logger.info(
        f"[Parameter Analysis]"
        f"[Full Model:{complexity['full_params_m']:.4f} M]"
        f"[Inference:{complexity['params_m']:.4f} M]"
        f"[Training-only:{complexity['excluded_params_m']:.4f} M]"
    )

    if complexity['unsupported_ops']:
        logger.warning(
            "[Unsupported FLOP operators] "
            f"{complexity['unsupported_ops']}"
        )

    logger.info("=" * 80)

    return complexity
