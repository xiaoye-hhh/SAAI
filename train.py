import logging
import os
import pprint

import torch
import yaml
# from apex import amp
from torch import optim

from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from models.baseline import Baseline


def train(cfg): 
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + cfg.log_name,
                        filemode="w")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(pprint.pformat(cfg))

    # training data loader
    train_loader = get_train_loader(dataset=cfg.dataset,
                                    root=cfg.data_root,
                                    sample_method=cfg.sample_method,
                                    batch_size=cfg.batch_size,
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    image_size=cfg.image_size,
                                    num_workers=8)

    # evaluation data loader
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)

    # model
    model = Baseline(num_classes=cfg.num_id,
                     backbone=cfg.backbone,
                     pattern_attention=cfg.pattern_attention,
                     modality_attention=cfg.modality_attention,
                     mutual_learning=cfg.mutual_learning,
                     drop_last_stride=cfg.drop_last_stride,
                     triplet=cfg.triplet,
                     k_size=cfg.k_size,
                     center_cluster=cfg.center_cluster,
                     center=cfg.center,
                     margin=cfg.margin,
                     num_parts=cfg.num_parts,
                     weight_KL=cfg.weight_KL,
                     weight_sid=cfg.weight_sid,
                     weight_sep=cfg.weight_sep,
                     update_rate=cfg.update_rate,
                     classification=cfg.classification,
                     margin1 = cfg.margin1,
                     margin2 = cfg.margin2,
                     dp = cfg.dp,
                     dp_w = cfg.dp_w,
                     cs_w = cfg.cs_w)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(model))

    model.cuda()

    # optimizer
    # assert cfg.optimizer in ['adam', 'sgd']
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    # else:
    # optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True)
    # ignored_params = list(map(id, model.local_conv_list.parameters())) \
    #                     + list(map(id, model.fc_list.parameters())) \
    #                     + list(map(id, model.attention_pool.parameters()))
        
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    # optimizer = optim.SGD([
    #         {'params': base_params, 'lr': cfg.lr * 0.05},
    #         {'params': model.local_conv_list.parameters(), 'lr': cfg.lr},
    #         {'params': model.fc_list.parameters(), 'lr': cfg.lr},
    #         {'params': model.attention_pool.parameters(), 'lr': cfg.lr}
    #         ],
    # weight_decay=5e-4, momentum=0.9, nesterov=True)

    # convert model for mixed precision training
    # model, optimizer = amp.initialize(model, optimizer, enabled=cfg.fp16, opt_level="O1")
    # if cfg.center:
    #     model.center_loss.centers = model.center_loss.centers.float()
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                                               milestones=cfg.lr_step,
    #                                               gamma=0.1)

    def step_lr_with_warmup(epoch):
        if epoch < 10:
            return (epoch + 1) / 10 
        else:
            if epoch < cfg.lr_step[0]:
                return 1
            elif epoch < cfg.lr_step[1]:
                return 0.1
            else:
                return 0.01
                
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=step_lr_with_warmup)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint)

    # engine
    checkpoint_dir = os.path.join("checkpoints", cfg.dataset, cfg.prefix)
    engine = get_trainer(dataset=cfg.dataset,
                         model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         start_eval=cfg.start_eval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)


if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/SYSU.yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_name", type=str, default="log.txt")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--update_rate", type=float, default=0.02)
    parser.add_argument("--num_parts", type=int, default=7)
    parser.add_argument("--margin1", type=float, default=0.01)
    parser.add_argument("--margin2", type=float, default=0.7)
    parser.add_argument("--dp", type=str, default="l2")
    parser.add_argument("--dp_w", type=float, default=0.5)
    parser.add_argument("--cs_w", type=float, default=1)
    args = parser.parse_args()

    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # enable cudnn backend
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    if cfg.sample_method == 'identity_uniform':
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.log_name = args.log_name
    cfg.backbone = args.backbone
    cfg.update_rate = args.update_rate
    cfg.num_parts = args.num_parts
    cfg.prefix = f"{cfg.prefix}_{cfg.log_name}"
    cfg.margin1 = args.margin1
    cfg.margin2 = args.margin2
    cfg.dp = args.dp
    cfg.dp_w = args.dp_w
    cfg.cs_w = args.cs_w
    cfg.freeze()

    train(cfg)
