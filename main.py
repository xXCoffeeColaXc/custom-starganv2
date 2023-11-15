import os
import argparse
from solver import Solver
from stargan import Stargan
from config import Config
from data_loader import get_loader, get_mask_loader
from torch.backends import cudnn
from utils import *
import wandb

def main(config):
    # For fast training.
    cudnn.benchmark = True

    create_folders(config=config)

    # Data loader.
    dataloader = None

    dataloader = get_loader(config.image_dir, ['daytime'],
                            config.image_size, config.batch_size,
                            config.mode, config.num_workers)
    
    # mask_dataloader = get_mask_loader(config.mask_image_dir, config.selected_attrs,
    #                         config.image_size, config.batch_size,
    #                         config.mode, config.num_workers)
    
    ref_dataloader = get_loader(config.image_dir, config.selected_attrs,
                            config.image_size, config.batch_size,
                            config.mode, config.num_workers)
    
    ref2_dataloader = get_loader(config.image_dir, config.selected_attrs,
                            config.image_size, config.batch_size,
                            config.mode, config.num_workers)
    

    config_object = Config(config=config)
    # Solver for training and testing StarGAN.
    model = Stargan(config=config_object, dataloader=dataloader, ref_dataloader=ref_dataloader, ref2_dataloader=ref2_dataloader)
    solver = Solver(model=model, config=config_object)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    if config.wandb:
        # Finish the WandB run when you're done training
        wandb.finish() # NOTE reallocate this
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--nd', type=int, default=3, help='dimension of domain labels')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--nf', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--nz', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--sdim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_ds', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=1, help='weight for cycle consistency loss')
    parser.add_argument('--lambda_gp', type=float, default=1, help='weight for gradient penalty')
    parser.add_argument('--lambda_style', type=float, default=1, help='weight for style penalty')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=200, help='')
    parser.add_argument('--ds_epochs', type=int, default=5, help='')
    parser.add_argument('--weight_decay', type=int, default=0.0001, help='weight decay')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--f_lr', type=float, default=0.000001, help='learning rate for F')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes/domains', default=['daytime', 'night'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--wandb', type=int, choices=[0, 1], help='enable wandb logging (0 for False, 1 for True)')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # TODO add val

    # Directories.
    parser.add_argument('--image_dir', type=str, default='/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/rgb_anon_trainvaltest/rgb_anon')
    parser.add_argument('--mask_image_dir', type=str, default='/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/gt_trainval/gt')
    parser.add_argument('--log_dir', type=str, default='outputs/logs')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--validation_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)