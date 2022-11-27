class Config():
    def __init__(self):
        self.batch_size = 8

        self.num_gen_blocks = 2
        self.num_dis_blocks = 4

        self.gen_in_channels = 128
        self.dis_in_channels = 64

        self.img_dims = (64, 64)

        self.noise_dims = 16

        # for optimizers
        self.lr = 0.003
        self.beta = 0.5

        self.epochs = 250

        self.gen_headstart = 0

        self.dropout = 0.0