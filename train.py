import torch
from torch import nn, optim

# for creating dataloader + augmentations
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image

import numpy as np

from config import Config
from utils import Generator, Discriminator


# training script - referenced DCGAN tutorial when neccesary
def train():
    config = Config()

    # creating dataloader to load from SSD
    dataset = datasets.ImageFolder(root="./data/pokemon_jpg",
                           transform=transforms.Compose([
                               transforms.Resize(config.img_dims),
                               transforms.CenterCrop(config.img_dims),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    generator = Generator(config)
    discriminator = Discriminator(config)

    # binary cross-entropy loss object declaration
    loss = nn.BCELoss()

    # declaring optimizer objects for each submodel (beta + lr values pulled from DCGAN tutorial)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=config.lr/2, betas=(config.beta, 0.999))
    gen_optimizer = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta, 0.999))


    # seeded input to verify generator training
    test_input = torch.rand(4, config.noise_dims**2)

    for epoch in range(config.epochs):
        if epoch % 5 == 0:
            # seeded test output
            with torch.no_grad():
                output = generator(test_input).detach()[0].numpy()
                # rescale back to proper img vals
                output += 0.5
                output *= 255

                image = Image.fromarray(np.transpose(np.uint8(output), (1, 2, 0)))
                image.save("output/epoch_" + str(epoch) + ".jpg")


        # iterate through samples produced by dataloader  
        for i, batch in enumerate(dataloader):
            if batch[0].shape[0] < 4:
                break
            
            # used in training processes for both generator and discriminator
            false_input = torch.rand(config.batch_size, config.noise_dims**2)
            false_gen_output = generator(false_input)

            total_dis_loss = 0

            # headstart for generator network training process to prevent diminishing gradients once discriminator converges
            if epoch >= config.gen_headstart:
                # DISCRIMINATOR TRAIN STEP


                # initialize gradients of discriminator to zero to begin training step
                discriminator.zero_grad()

                # train discriminator on real batch first
                true_labels = torch.ones((config.batch_size,1), dtype=torch.float)

                true_output = discriminator(batch[0])

                true_loss = loss(true_output, true_labels)
                true_loss.backward()

                # train discriminator on fake batch after
                false_labels = torch.zeros((config.batch_size,1), dtype=torch.float)

                # detach gen output to not train generator in this step
                false_dis_output = discriminator(false_gen_output.detach())

                false_loss = loss(false_dis_output, false_labels)
                false_loss.backward()

                # total discriminator loss
                total_dis_loss = true_loss + false_loss

                # only apply gradients to update discriminator weights
                dis_optimizer.step()


            # GENERATOR TRAIN STEP


            # initialize gradients of generator to zero to begin training step
            generator.zero_grad()

            # train generator on mispredicted discriminator labels
            gen_labels = torch.ones((config.batch_size,1), dtype=torch.float)

            # train generator to trick discriminator
            train_gen_dis_output = discriminator(false_gen_output)

            gen_loss = loss(train_gen_dis_output, gen_labels)
            gen_loss.backward()

            # only apply gradients to update generator weights
            gen_optimizer.step()

            if i % 50 == 0:
                if epoch >= config.gen_headstart:
                    print("Discriminator loss: " + str(torch.mean(total_dis_loss)))
                print("Generator loss: " + str(torch.mean(gen_loss)))
        


if __name__ == '__main__':
    train()