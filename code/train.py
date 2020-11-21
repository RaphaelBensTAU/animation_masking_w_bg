from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater

def train(config, generator, mask_generator,bg_generator,bg_refiner, checkpoint, log_dir, dataset, device_ids, without_bg = False):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_mask_generator = torch.optim.Adam(mask_generator.parameters(), lr=train_params['lr_mask_generator'], betas=(0.5, 0.999))

    if not without_bg:
        optimizer_bg_generator = torch.optim.Adam(bg_generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
        optimizer_bg_refiner = torch.optim.Adam(bg_refiner.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        print('loading cpk')
        start_epoch = Logger.load_cpk(checkpoint, generator, mask_generator, optimizer_generator, optimizer_mask_generator, bg_generator, optimizer_bg_generator, bg_refiner, optimizer_bg_refiner)
        start_epoch += 1
    else:
        start_epoch = 0

    print(start_epoch)
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_mask_generator = MultiStepLR(optimizer_mask_generator, train_params['epoch_milestones'], gamma=0.1, last_epoch=-1 + start_epoch * (train_params['lr_mask_generator'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, 150)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(mask_generator, generator, bg_generator,bg_refiner, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for index, x in enumerate(dataloader):
                predict_mask = epoch >= 1 #real is 1 
                predict_bg = epoch >= 2 if not without_bg else False

                losses_generator, bg_losses, generated = generator_full(x, predict_mask, predict_bg)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                bg_loss_values = [val.mean() for val in bg_losses.values()]
                loss_bg = sum(bg_loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()

                optimizer_mask_generator.step()
                optimizer_mask_generator.zero_grad()

                if predict_bg:
                        loss_bg.backward()

                        optimizer_bg_generator.step()
                        optimizer_bg_generator.zero_grad()

                        optimizer_bg_refiner.step()
                        optimizer_bg_refiner.zero_grad()

                print("EPOCH {}, BATCH {}".format(epoch,index))
                print(losses_generator.items())
                print(bg_losses)

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_mask_generator.step()

            logger.log_epoch(epoch, {'generator': generator,
                                     'mask_generator': mask_generator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_mask_generator': optimizer_mask_generator,
                                     'bg_generator' : bg_generator,
                                     'bg_refiner' : bg_refiner,
                                     'optimizer_bg_generator' : optimizer_bg_generator,
                                     'optimizer_bg_refiner' : optimizer_bg_refiner} , inp=x, out=generated, save_w=True)
