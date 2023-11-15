import os
import torch
import torch.nn as nn
from modules import *
import wandb

class Stargan(object):
    def __init__(self, config, dataloader, ref_dataloader, ref2_dataloader) -> None:
        self.dataloader = dataloader
        self.ref_dataloader = ref_dataloader
        self.ref2_dataloader = ref2_dataloader
        self.config = config
        
        # Build the model and tensorboard.
        self.build_model()

        if self.config.mode == 'train':
            self.train()
        elif self.config.mode == 'val':
            self.eval()

        if self.config.wandb:
            print(self.config.wandb)
            self.setup_logger()


    def build_model(self):
        """Create a generator and a discriminator."""
        
        self.G = Generator(self.config.nf, self.config.sdim)
        self.F = MappingNetwork(self.config.nz, self.config.nd, self.config.sdim)
        self.D = Discriminator(self.config.nf, self.config.nd)
        self.E = StyleEncoder(self.config.nf, self.config.nd, self.config.sdim)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.config.g_lr, betas=[self.config.beta1, self.config.beta2], weight_decay=self.config.weight_decay)
        self.f_optimizer = torch.optim.Adam(self.F.parameters(), lr=self.config.f_lr, betas=[self.config.beta1, self.config.beta2], weight_decay=self.config.weight_decay)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.config.d_lr, betas=[self.config.beta1, self.config.beta2], weight_decay=self.config.weight_decay)
        self.e_optimizer = torch.optim.Adam(self.E.parameters(), lr=self.config.g_lr, betas=[self.config.beta1, self.config.beta2], weight_decay=self.config.weight_decay)

        self.l1 = nn.L1Loss()

        self.print_network(self.G, 'G')
        self.print_network(self.F, 'F')
        self.print_network(self.D, 'D')
        self.print_network(self.E, 'E')
            
        self.G.to(self.config.device)
        self.F.to(self.config.device)
        self.D.to(self.config.device)
        self.E.to(self.config.device)

    def save_model(self, num_iter):
        G_path = os.path.join(self.config.model_save_dir, '{}-G.ckpt'.format(num_iter))
        F_path = os.path.join(self.config.model_save_dir, '{}-F.ckpt'.format(num_iter))
        D_path = os.path.join(self.config.model_save_dir, '{}-D.ckpt'.format(num_iter))
        E_path = os.path.join(self.config.model_save_dir, '{}-E.ckpt'.format(num_iter))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.F.state_dict(), F_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.E.state_dict(), E_path)
        print('Saved config checkpoints into {}...'.format(self.config.model_save_dir))


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    # TODO F, E
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.config.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.config.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def train(self):
        self.G.train()
        self.F.train()
        self.D.train()
        self.E.train()

    def eval(self):
        self.G.eval()
        self.F.eval()
        self.D.eval()
        self.E.eval()

    def setup_logger(self):
            # Initialize WandB
            wandb.init(project='custom-starganv2-weather', entity='tamsyandro', config={
                "d_lr": self.config.d_lr, 
                "g_lr": self.config.g_lr,
                "f_lr": self.config.g_lr,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "image_size": self.config.image_size,
                "selected_domains": self.config.selected_attrs,
                "lambda_ds": self.config.lambda_ds,
                "lambda_rec": self.config.lambda_rec,
                "lambda_gp": self.config.lambda_gp,
                "lambda_style": self.config.lambda_style,
                "n_critic": self.config.n_critic,
                "weight_decay": self.config.weight_decay,
               
                # ... Add other hyperparameters here
            })

            # Ensure DEVICE is tracked in WandB
            wandb.config.update({"device": self.config.device})

