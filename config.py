import torch


class Config(object):
    def __init__(self, config) -> None:
        # Model configurations.
        self.nd = config.nd
        self.image_size = config.image_size
        self.nf = config.nf
        self.nz = config.nz
        self.sdim = config.sdim
        self.lambda_ds = config.lambda_ds
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_style = config.lambda_style

        # Training configurations.
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.ds_epochs = config.ds_epochs
        self.weight_decay = config.weight_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.f_lr = config.f_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters
        self.wandb = config.wandb
        self.mode = config.mode

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.image_dir = config.image_dir
        self.mask_image_dir = config.mask_image_dir
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.validation_step = config.validation_step
        self.model_save_step = config.model_save_step



    def get_config(self):
        pass

    def update_config(self):
        pass