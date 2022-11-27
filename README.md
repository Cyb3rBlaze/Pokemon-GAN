# Pokemon GAN!!!

## Notes:
- Used dropout + reduced size of discriminator to prevent unbalanced convergence
- Created headstart pipeline to initially train generator params to also help with convergence problem
- Needed larger learning rate to help generator converge faster
- ConvTranspose2d worked better than Upsample + Conv2d
- Make sure to check normalization function!
- Referenced: https://blog.jovian.ai/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d