import torch
from torch import nn, optim

# for creating dataloader + augmentations
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.utils import save_image

from config import Config
from utils import Generator, Discriminator


device = torch.device("cuda")

normalization_stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# for saving image purposes
def denorm(image):
    return image * normalization_stats[1][0] + normalization_stats[0][0]


# training script - referenced DCGAN tutorial when neccesary
def train():
    config = Config()

    # creating dataloader to load from SSD
    normal_dataset = datasets.ImageFolder("./data/pokemon_jpg", transform=transforms.Compose([
        transforms.Resize(config.img_dims),
        transforms.CenterCrop(config.img_dims),
        transforms.ToTensor(),
        transforms.Normalize(*normalization_stats)]))

    # Augment the dataset with mirrored images
    mirror_dataset = datasets.ImageFolder("./data/pokemon_jpg", transform=transforms.Compose([
        transforms.Resize(config.img_dims),
        transforms.CenterCrop(config.img_dims),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(*normalization_stats)]))

    # Augment the dataset with color changes
    color_jitter_dataset = datasets.ImageFolder("./data/pokemon_jpg", transform=transforms.Compose([
        transforms.Resize(config.img_dims),
        transforms.CenterCrop(config.img_dims),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        transforms.Normalize(*normalization_stats)]))

    # Combine the datasets
    dataset_list = [normal_dataset, mirror_dataset, color_jitter_dataset]
    dataset = ConcatDataset(dataset_list)

    dataloader = DataLoader(dataset, config.batch_size, shuffle=True, pin_memory=False)


    generator = Generator(config)
    generator.to(device)
    discriminator = Discriminator(config)
    discriminator.to(device)

    # binary cross-entropy loss object declaration
    loss = nn.BCELoss()

    # declaring optimizer objects for each submodel (beta + lr values pulled from DCGAN tutorial)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta, 0.9))
    gen_optimizer = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta, 0.9))


    # seeded input to verify generator training
    test_input = torch.rand(1, config.noise_dims, 1, 1).to(device)

    for epoch in range(config.epochs):
        print("EPOCH: " + str(epoch))
        if epoch % 5 == 0:
            # seeded test output
            with torch.no_grad():
                output = generator(test_input)

                save_image(denorm(output), "output/epoch_" + str(epoch) + ".jpg")


        # iterate through samples produced by dataloader  
        for i, batch in enumerate(dataloader):
            total_dis_loss = 0


            # DISCRIMINATOR TRAIN STEP


            # initialize gradients of discriminator to zero to begin training step
            dis_optimizer.zero_grad()

            # train discriminator on real batch first
            true_labels = (torch.rand(batch[0].shape[0], 1, device=device) * (0.1 - 0) + 0).to(device)

            true_output = discriminator((batch[0]).to(device))

            true_loss = loss(true_output, true_labels)

            # train discriminator on fake batch after - contains some noisy data to throw of discriminator
            false_labels = (torch.rand(batch[0].shape[0], 1, device=device) * (1 - 0.9) + 0.9).to(device)

            # training on generator output
            false_input = torch.rand(batch[0].shape[0], config.noise_dims, 1, 1).to(device)
            false_gen_output = generator(false_input)
            false_dis_output = discriminator(false_gen_output)

            false_loss = loss(false_dis_output, false_labels)

            # total discriminator loss
            total_dis_loss = true_loss + false_loss
            total_dis_loss.backward()

            # only apply gradients to update discriminator weights
            dis_optimizer.step()


            # GENERATOR TRAIN STEP


            # initialize gradients of generator to zero to begin training step
            gen_optimizer.zero_grad()

            # train generator on mispredicted discriminator labels
            gen_labels = torch.zeros((config.batch_size,1), dtype=torch.float).to(device)

            # train generator to trick discriminator
            false_input = torch.rand(config.batch_size, config.noise_dims, 1, 1).to(device)
            false_gen_output = generator(false_input)
            train_gen_dis_output = discriminator(false_gen_output)

            gen_loss = loss(train_gen_dis_output, gen_labels)
            gen_loss.backward()

            # only apply gradients to update generator weights
            gen_optimizer.step()


            if i % 200 == 0:
                print("Discriminator loss: " + str(torch.mean(total_dis_loss)))
                print("Generator loss: " + str(torch.mean(gen_loss)))
        

if __name__ == '__main__':
    train()