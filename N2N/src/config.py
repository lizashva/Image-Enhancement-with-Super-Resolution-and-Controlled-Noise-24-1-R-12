

class TrainConfig:
    def __init__(self, noise_type, noise_param):
        self.train_dir = './data/train'
        self.train_size = 500
        self.valid_dir = './data/valid'
        self.valid_size = 100
        self.ckpt_save_path = './ckpts'
        self.ckpt_overwrite = False
        self.report_interval = 25
        self.nb_epochs = 30
        self.loss = 'l2'
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.crop_size = 128
        self.plot_stats = False
        self.cuda = True

class TestConfig:
    def __init__(self, noise_type, noise_param):
        self.data = './data/test'
        self.load_ckpt = self.set_load_ckpt(noise_type)
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.seed = 1
        self.crop_size = 256
        self.show_output = 2

    def set_load_ckpt(self, noise_type):
        filename_noise_type = noise_type if noise_type != 'textRemoval' else 'text'

        # Generate the checkpoint path string
        checkpoint_path = f'./ckpts/{filename_noise_type}/n2n-{filename_noise_type}.pt'
        return checkpoint_path


def set_configuration(self, noise_type, noise_param, trainable):
    valid_noise_types = ['gaussian', 'poisson', 'textRemoval']
    if noise_type not in valid_noise_types:
        raise ValueError(f"Unsupported noise type '{noise_type}'. Choose from {valid_noise_types}.")

    if trainable:
        train_config = TrainConfig(noise_type, noise_param)
    else:
        test_config = TestConfig(noise_type, noise_param)