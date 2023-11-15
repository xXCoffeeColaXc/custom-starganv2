from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import os
import time
import datetime
from utils import *
import wandb

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, model, config):
        self.model = model # contains dataloader, G, D and optimizers
        self.config = config # contains config data

    # def update_lr(self, g_lr, d_lr):
    #     """Decay learning rates of the generator and discriminator."""
    #     for param_group in self.model.g_optimizer.param_groups:
    #         param_group['lr'] = g_lr
    #     for param_group in self.model.d_optimizer.param_groups:
    #         param_group['lr'] = d_lr

    #     self.config.g_lr = g_lr # TODO test this
    #     self.config.d_lr = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.model.g_optimizer.zero_grad()
        self.model.d_optimizer.zero_grad()
        self.model.f_optimizer.zero_grad()
        self.model.e_optimizer.zero_grad()

    # def gradient_penalty(self, y_orig, x_orig):
    #     """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    #     weight = torch.ones(y_orig.size()).to(self.config.device)
    #     dydx = torch.autograd.grad(outputs=y_orig,
    #                                inputs=x_orig,
    #                                grad_outputs=weight,
    #                                retain_graph=True,
    #                                create_graph=True,
    #                                only_inputs=True)[0]

    #     dydx = dydx.view(dydx.size(0), -1)
    #     dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    #     return torch.mean((dydx_l2norm-1)**2)
    
    def gradient_penalty(self, out, x_orig):
        grad = torch.autograd.grad(
            outputs=out.sum(),
            inputs=x_orig,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0].view(x_orig.size(0), -1)
        return 0.5 * torch.mean(torch.sum(grad.pow(2), dim=1))
    
    def preprocess_input_data(self, x_real, label_org):
        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = label2onehot(label_org, self.config.c_dim)
        c_trg = label2onehot(label_trg, self.config.c_dim)

        x_real = x_real.to(self.config.device)           # Input images.
        c_org = c_org.to(self.config.device)             # Original domain labels.
        c_trg = c_trg.to(self.config.device)             # Target domain labels.
        label_org = label_org.to(self.config.device)     # Labels for computing classification loss.
        label_trg = label_trg.to(self.config.device)     # Labels for computing classification loss.

        return x_real, c_org, c_trg, label_org, label_trg

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target) # This could be passed as CrossEntropy()
    
    def adv_loss(logits, target):
        # NOTE better way thats for sure, find a way to use it.
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss
    
    def compute_d_loss(self, x_real, label_org, c_trg):
        # Compute loss with real images.
        out_src, out_cls = self.model.D(x_real)
        d_loss_real = - torch.mean(out_src) # loss_real = adv_loss(out, 1)
        d_loss_cls = self.classification_loss(out_cls, label_org)

        # Compute loss with fake images.
        x_fake = self.model.G(x_real, c_trg)
        out_src, out_cls = self.model.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src) # loss_fake = adv_loss(out, 0)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.config.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.model.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls + self.config.lambda_gp * d_loss_gp
    
        # NOTE best practice: return d_loss, Object(d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp)
        return d_loss, d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp

    def compute_g_loss(self, x_real, c_org, label_trg, c_trg):
        # Original-to-target domain.
        x_fake = self.model.G(x_real, c_trg)
        out_src, out_cls = self.model.D(x_fake)
        g_loss_fake = - torch.mean(out_src) # loss_adv = adv_loss(out, 1)
        g_loss_cls = self.classification_loss(out_cls, label_trg)

        # Target-to-original domain.
        x_reconst = self.model.G(x_fake, c_org)
        g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) # This is just an L1 loss
        # TODO try out feeding this loss to discriminator instead of generator,
        # cycle loss will be erased ? 

        # Backward and optimize.
        g_loss = g_loss_fake + self.config.lambda_rec * g_loss_rec + self.config.lambda_cls * g_loss_cls
        
        return g_loss, g_loss_fake, g_loss_rec, g_loss_cls

    def train(self):
        """Train StarGAN within a single dataset."""

        start_epoch = 0

        lambda_ds_zero = self.config.lambda_ds
        self.config.lambda_ds = max((lambda_ds_zero * (self.config.ds_epochs - start_epoch)) / self.config.epochs, 0.)


        # Set data loader.
        dataloader = self.model.dataloader
        ref_dataloader = self.model.ref_dataloader
        ref2_dataloader = self.model.ref2_dataloader

        # Start training.
        print('Start training...')
        # TODO tqdm and duration for whole epoch
        for ep in range(start_epoch, self.config.epochs):
            start_time = time.time() # duration for one epoch
            for batch_idx, ((x_orig, y_orig), (x_ref, y_target), (x_ref2, _)) in enumerate(zip(dataloader, ref_dataloader, ref2_dataloader)):
      
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

                x_orig = x_orig.to(self.config.device)
                x_orig.requires_grad_()
                y_orig = y_orig.to(self.config.device).squeeze(1)
                x_ref = x_ref.to(self.config.device)
                y_target = y_target.to(self.config.device).squeeze(1)
                x_ref2 = x_ref2.to(self.config.device)
                z = torch.randn((x_orig.size(0), self.config.nz)).to(self.config.device) # random latent code from normal dist

            # =================================================================================== #
            #                             2. Train the discriminator with z                       #
            # =================================================================================== #

                self.reset_grad()

                out = self.model.D(x_orig, y_orig)
                real_label = torch.full_like(out, fill_value=1)
                d_loss_real = F.binary_cross_entropy_with_logits(out, real_label)
                d_loss_gp = self.gradient_penalty(out, x_orig)

                with torch.no_grad():
                    s_ = self.model.F(z, y_target)
                    x_fake = self.model.G(x_orig, s_)
                
                out = self.model.D(x_fake, y_target)
                fake_label = torch.full_like(out, fill_value=0)
                d_loss_fake = F.binary_cross_entropy_with_logits(out, fake_label)

                d_loss = d_loss_real + d_loss_fake + d_loss_gp * self.config.lambda_gp

                d_loss.backward()
                self.model.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/d_loss'] = d_loss.item()

            # =================================================================================== #
            #                             3. Train the discriminator with x_ref                   #
            # =================================================================================== #

                self.reset_grad()

                out = self.model.D(x_orig, y_orig)
                real_label = torch.full_like(out, fill_value=1)
                d_loss_real = F.binary_cross_entropy_with_logits(out, real_label)
                d_loss_gp = self.gradient_penalty(out, x_orig)

                with torch.no_grad():
                    s_ = self.model.E(x_ref, y_target)
                    x_fake = self.model.G(x_orig, s_)
                
                out = self.model.D(x_fake, y_target)
                fake_label = torch.full_like(out, fill_value=0)
                d_loss_fake = F.binary_cross_entropy_with_logits(out, fake_label)

                d_loss = d_loss_real + d_loss_fake + d_loss_gp * self.config.lambda_gp

                d_loss.backward()
                self.model.d_optimizer.step()

                # Logging.
                loss['D2/d_loss'] = d_loss.item()
            
            # =================================================================================== #
            #                               4. Train the generator with z, z2                     #
            # =================================================================================== #
            
                # if (batch_idx+1) % self.config.n_critic == 0:
                self.reset_grad()

                s_ = self.model.F(z, y_target)
                x_fake = self.model.G(x_orig, s_)
                out = self.model.D(x_fake, y_target)
                real_label = torch.full_like(out, fill_value=1)
                g_loss_adv = F.binary_cross_entropy_with_logits(out, real_label)

                s_pred = self.model.E(x_fake, y_target)
                g_loss_style = self.model.l1(s_, s_pred)

                z2 = torch.randn((x_orig.size(0), self.config.nz)).to(self.config.device)
                s_2 = self.model.F(z2, y_target)
                x_2 = self.model.G(x_fake, s_2)
                g_loss_ds = self.model.l1(x_fake, x_2.detach())

                s = self.model.E(x_orig, y_orig)
                x_rec = self.model.G(x_fake, s)
                g_loss_rec = self.model.l1(x_orig, x_rec)

                g_loss = g_loss_adv + self.config.lambda_style * g_loss_style - self.config.lambda_ds * g_loss_ds + self.config.lambda_rec * g_loss_rec

                g_loss.backward()
                self.model.g_optimizer.step()
                self.model.f_optimizer.step()
                self.model.e_optimizer.step()

                # Logging.
                loss['G/g_loss'] = g_loss.item()


            # =================================================================================== #
            #                               5. Train the generator with x_ref, x_ref2             #
            # =================================================================================== #
            
                self.reset_grad()

                s_ = self.model.E(x_ref, y_target)
                x_fake = self.model.G(x_orig, s_)
                out = self.model.D(x_fake, y_target)
                real_label = torch.full_like(out, fill_value=1)
                g_loss_adv = F.binary_cross_entropy_with_logits(out, real_label)

                s_pred = self.model.E(x_fake, y_target)
                g_loss_style = self.model.l1(s_, s_pred)

                s_2 = self.model.E(x_ref2, y_target)
                x_2 = self.model.G(x_fake, s_2)
                g_loss_ds = self.model.l1(x_fake, x_2.detach())

                s = self.model.E(x_orig, y_orig)
                x_rec = self.model.G(x_fake, s)
                g_loss_rec = self.model.l1(x_orig, x_rec)

                g_loss = g_loss_adv + self.config.lambda_style * g_loss_style - self.config.lambda_ds * g_loss_ds + self.config.lambda_rec * g_loss_rec

                g_loss.backward()
                self.model.g_optimizer.step()

                # Logging.
                loss['G2/g_loss'] = g_loss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

                self.config.lambda_ds -= lambda_ds_zero / (len(dataloader) * self.config.epochs)
                self.config.lambda_ds = max(self.config.lambda_ds, 0.)

                num_iter = batch_idx + ep * len(dataloader) + 1

                # Print out training information.
                if num_iter % self.config.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, batch_idx+1, len(dataloader))
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                # # Calculate and Log validation loss
                # if (i+1) % self.config.validation_step == 0: # TODO Validation step
                #     self.validate()
                
                if num_iter % self.config.sample_step == 0:
                    self.model.eval()
                    with torch.no_grad():
                        concatenated_images = []
                        (x_orig, _), (x_ref, y_target) = next(iter(zip(dataloader, ref_dataloader))) # TODO testloader
                        x_orig = x_orig.to(self.config.device)
                        x_ref = x_ref.to(self.config.device)
                        y_target = y_target.to(self.config.device).squeeze(1)
                        s_ = self.model.E(x_ref, y_target)
                        x_fake = self.model.G(x_orig, s_)
                        
                        # Concatenate x_orig, x_ref, x_fake horizontally
                        for idx in range(self.config.batch_size):
                            concatenated_row = torch.cat((x_orig[idx], x_ref[idx], x_fake[idx]), dim=2)  # Concatenate along width
                            concatenated_images.append(concatenated_row)
        

                        # Stack all concatenated rows vertically
                        final_image = torch.cat(concatenated_images, dim=1)  # Concatenate along height

                        # Save the final image
                        sample_path = os.path.join(self.config.sample_dir, '{}-images.jpg'.format(num_iter))
                        save_image(final_image.cpu(), sample_path, nrow=1, normalize=True)

                    self.model.train()
                    print("Iter %d: Generate images for test dataset" % (num_iter))


                # Save config checkpoints.
                if num_iter % self.config.model_save_step == 0:
                    self.model.save_model(num_iter)

                if self.config.wandb:
                    # Log to wandb
                    wandb.log(loss)
                    wandb.log({
                        "lambda_ds": self.config.lambda_ds,
                        "num_iters": num_iter,
                        "epochs": ep,
                        })
          
    def validate(self):
        # Get a batch of validation data
        # NOTE just a batch, all do I need to iter through all of the val data?
        val_loader = self.model.val_loader

        self.model.D.eval()
        self.model.G.eval()
    
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for i, (x_real, label_org) in enumerate(val_loader):

                x_real, c_org, c_trg, label_org, label_trg = self.preprocess_input_data(x_real, label_org)

                g_loss, _, _, _ = self.compute_g_loss(x_real, c_org, label_trg, c_trg)

                # Aggregate val_loss through one batch
                total_val_loss += g_loss.item()
                num_batches += 1

        # Calculate mean
        average_val_loss = total_val_loss / num_batches

        self.model.D.train()
        self.model.G.train()

        if self.config.wandb:
            # Log loss
            wandb.log({"val_loss": average_val_loss})

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.model.restore_config(self.config.test_iters)
        
        data_loader = self.model.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.config.device)
                c_trg_list = create_labels(c_org=c_org, c_dim=self.config.c_dim, config=self.config)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.model.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.config.result_dir, '{}-images.jpg'.format(i+1))
                save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

