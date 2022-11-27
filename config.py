class Config():
    def __init__(self):
        self.batch_size = 4

        self.num_gen_blocks = 3
        self.num_dis_blocks = 5

        self.gen_in_channels = 128
        self.dis_in_channels = 64

        self.img_dims = (128, 128)

        self.noise_dims = 100

        # for optimizers
        self.lr = 0.0002
        self.beta = 0.5

        self.epochs = 50