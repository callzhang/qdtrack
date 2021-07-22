from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    with open(opt.data_cfg) as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']

    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print(f'Train on epoch: {epoch}')
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (opt.lr_decay ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()

'''
DETRAC_MOT20_KITTI: ctdet_coco_dla_2x, batch=4, lr=1e-4
DETRAC_MOT20_KITTI_fairmot: fairmot_dla34.pth, batch=16, lr=5e-4
'''

if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse(
        [
            '--exp_id', 'DETRAC_MOT20_KITTI_2',#updated dataset
            '--data_dir', 'kitti_tracking',
            '--data_cfg', 'src/lib/cfg/mot20_kitti_detrac.json',
            '--load_model', 'models/ctdet_coco_dla_2x.pth',
            # '--load_model', 'models/fairmot_dla34.pth',
            # '--load_model', 'exp/mot/DETRAC_MOT20_KITTI/model_15.pth',
            '--num_classes', '8',
            # '--ltrb', False,
            '--batch_size', '4',
            '--track_buffer', '150',
            '--dataset', 'kitti',
            '--lr', '1e-4',
            '--lr_step', '5,10,15,20,25',
            # '--resume',
    ])
    # train yolo
    # opt = opts().parse([
    #     '--exp_id', 'DETRAC_MOT20_KITTI',
    #     '--data_cfg', 'src/lib/cfg/mot20_kitti_detrac.json',
    #     '--load_model', 'models/ctdet_coco_dla_2x.pth',
    #     '--lr', '1e-4',
    #     '--batch_size', '4',
    #     '--wh_weight', '0.2',
    #     '--multi_loss', 'fix',
    #     '--arch', 'yolo',
    #     '--reid_dim', '64'
    # ])
    main(opt)
