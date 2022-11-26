class Config():
    def __init__(self):
        self.batch_size = 4

        self.num_gen_upscale_blocks = 2
        self.num_gen_const_blocks = 2
        self.num_dis_blocks = 2
        self.dis_hidden_units = 48

        self.gen_in_channels = 8
        self.dis_in_channels = 8

        self.img_dims = (64, 64)

        self.noise_dims = 8

        self.flatten_dims = ((self.img_dims[0]/(2**self.num_dis_blocks))**2)*((8*2)*(self.num_dis_blocks-1))

        # for optimizers
        self.lr = 0.0002
        self.beta = 0.5

        self.epochs = 50

        self.gen_headstart = 5

        self.dropout = 0.9