# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
import Config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D,ImageToImage2Deasy
from nets.LViT import LViT

from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile
import nets.vit_seg_configs as transconfig_vit
from nets.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.vit_seg_modeling_txt import VisionTransformer as ViT_seg_txt
import shutil
def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)
def save_lastcheckpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
 
 
    model = state['model']  # model type


    filename = save_path + '/' + \
               'last_model-{}.pth.tar'.format(model)

    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2Deasy(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'Covid19':
        train_text = read_text(config.train_dataset + 'Train_text_for_Covid19.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text_for_Covid19.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2Deasy(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'MosMed':
        train_text = read_text(config.train_dataset + 'Train_text_for_MosMed.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text_for_MosMed.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2Deasy(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)


    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        pretrained_UNet_model_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')





    elif model_type == config.model_name:
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))

        transconfigvit = transconfig_vit.get_r50_b16_config()
        model = ViT_seg_txt(transconfigvit, img_size=224, num_classes=1)
        sam_path = "/data1/Code/zhaoxiaowei/RecLMIS-main/model/sam_vit_b_01ec64.pth"
        sam_state_dict = torch.load(sam_path)
        print(sam_path)
        model.sam_encoder_output.load_state_dict(sam_state_dict, strict=False)
 
        # # 2. Freeze everything
        # for param in model.sam_encoder_output.parameters():
        #     param.requires_grad = False
 
           
        # # 遍历每个 block，只解冻其中的 Adapter
        # for blk in model.sam_encoder_output.blocks:
        #     if hasattr(blk, 'Space_Adapter'):
        #         for p in blk.Space_Adapter.parameters():
        #             p.requires_grad = True
        #     if hasattr(blk, 'MLP_Adapter'):
        #         for p in blk.MLP_Adapter.parameters():
        #             p.requires_grad = True
        #     if hasattr(blk, 'MMAdapter'):
        #         for p in blk.MMAdapter.parameters():
        #             p.requires_grad = True



    elif model_type == 'test':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))

        transconfigvit = transconfig_vit.get_r50_b16_config()
        model = ViT_seg_txt(transconfigvit, img_size=224, num_classes=1)
        model.load_from(
            weights=np.load("/data1/Code/zhaoxiaowei/LViT-Orignal/model/imagenet21k+imagenet2012_R50+ViT-B_16.npz"))
        sam_path = "/data1/Code/zhaoxiaowei/LViT-Orignal/model/sam_vit_b_01ec64.pth"
        sam_state_dict = torch.load(sam_path)
        print(sam_path)
        model.sam_encoder_output.load_state_dict(sam_state_dict, strict=False)
        model.maskdecoder.load_state_dict(sam_state_dict, strict=False)
    else:
        raise TypeError('Please enter a valid name for the model type')
    input = torch.randn(1, 3, 224, 224)
    text = torch.randn(1, 10, 768)
    flops, params = profile(model, inputs=(input, text))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lambda1 = lambda epoch: max(0.99**epoch, 0.1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
                logger.info('\t Mean dice:{:.4f} does not increase, '
                            'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
                # 保存当前模型为 last_model
                save_lastcheckpoint({
                    'epoch': epoch,
                    'best_model': False,
                    'model': model_type,
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict()
                }, config.model_path)
        # 🔽 新增逻辑：超过100 epoch，每5个epoch保存一次
        if (epoch + 1) >= 20 :
            ckpt_path = os.path.join(config.model_path, f'epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model': model_type,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict()
            }, ckpt_path)
            logger.info(f"\t Epoch {epoch + 1}: Checkpoint saved to {ckpt_path}")
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    import shutil
    suou = "/data1/Code/zhaoxiaowei/LViT-Orignal/nets/sammodels/common/adapter.py"

    destination_dir = os.path.join(config.save_path, 'adapter.py')

    shutil.copy2(suou, destination_dir)

    suou = "/data1/Code/zhaoxiaowei/LViT-Orignal/nets/vit_seg_modeling_txt.py"
    source_file = '/data1/Code/zhaoxiaowei/LViT-Orignal/train_model.py'
    source_file2 = '/data1/Code/zhaoxiaowei/LViT-Orignal/nets/sammodels/ImageEncoder/vit/adapter_block.py'
    source_file4 = '/data1/Code/zhaoxiaowei/LViT-Orignal/Train_one_epoch.py'
    source_file3 = '/data1/Code/zhaoxiaowei/LViT-Orignal/Config.py'
    destination_dir = os.path.join(config.save_path, 'vit_seg_modeling_txt.py')
    destination_dir1 = os.path.join(config.save_path, 'train_model.py')
    destination_dir2 = os.path.join(config.save_path, 'adapter_block.py')
    destination_dir3 = os.path.join(config.save_path, 'Config.py')
    destination_dir4 = os.path.join(config.save_path, 'Train_one_epoch.py')
    shutil.copy2(suou, destination_dir)
    shutil.copy2(source_file, destination_dir1)
    shutil.copy2(source_file2, destination_dir2)
    shutil.copy2(source_file3, destination_dir3)
    shutil.copy2(source_file4, destination_dir4)
    model = main_loop(model_type=config.model_name, tensorboard=True)
