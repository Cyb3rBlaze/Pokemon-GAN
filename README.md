# Pokemon GAN!!!

## Notes:
- Needed larger learning rate to help generator converge faster
- Upsample + Conv2d didn't have checkerboard artifacts present in ConvTranspose2d outputs
- Make sure to check normalization function for future implementations!
- Ensure that ALL the model paramaters are being trained (use nn.ModuleList to add submodules)