from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


# torch.manual_seed, torch.backends.cudnn.benchmark => randomness
# opt.seed = 317
# opt.not_cuda_benchmark -> action : store_true
# opt.text -> default : false , action : store_true
def main(opt):
    torch.manual_seed(opt.seed) 
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test 

    Dataset = get_dataset(opt.dataset, opt.task)
    # opt.dataset : default = coco
    # opt.task : default = ctdet

    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)
    
    # ops.gpus_str : 0 => 0 : multi-gpu / -1 : cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    
    # opt.arch = dla_34
    # opt.heads =
    # opt.head_conv = -1
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    # learning rate = 1.25e-4
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # opt.load_model = ctdet_coco_dla_2x.pth
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
            # opt.load_model = 파서에서 model path 입력해놓음
            # optimizer = Adam
            # opt.resume = default : false
            # opt.lr_step = 90,120

    Trainer = train_factory[opt.task]
    # opt.task : ctdet
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    # opt.gpus = default 0인데 0이 muilti gpu
    # opt.chunk_sizes = 
    # opt.device => opt.gpus가 0이상이면 cuda 아니면 cpu사용(BaseDetector 클래스 참고)

    print('setting up data...')
    val_loader = torch.utils.data.DataLoader( # validation dataloader 불러옴
        Dataset(opt, 'val'),
        batch_size = 1,
        shuffle=False,
        num_workers=1,
        pin_memory = True # True -> tensor를 cuda 고정 메로리에 올림
    )

    if opt.test: # opt.test = false
        _, preds = trainer.val(0, ValueError)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size, # 32
        shuffle=True, 
        num_workers=opt.num_workers, # 4
        pin_memory=True, # dataloader -> tensor를 CUDA 고정메모리로
        drop_last=True # 마지막에 남는 배치는 사용X
    )

    print('starting training')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # start_epoch = 0(min), opt.num_epoch = 140(max)
        mark = epoch if opt.save_all else 'last'
        # if opt.save_all => 5 epoch마다 한번씩 모델 저장
        
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(k, v))

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            # opt.val_intervals => default = 5 validation이 실행될때 epoch
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                        epoch, model, optimizer)
        
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} |'.format(k, v))
            if log_dict_val[opt.metric] < best:
                # opt.metric : loss(main metric to save best model)
                # best : 1e10
                best = log_dict_val[opt.metric]
                # opt.metric : loss 
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                            epoch, model)

        else: 
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                        epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step: # opt.lr_step = 90,120
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                        epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            # opt.lr = 1.25e-4
            print("drop lr to ", lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        logger.close()

    if __name__ == '__main__':
        opt = opts().parse()
        main(opt)