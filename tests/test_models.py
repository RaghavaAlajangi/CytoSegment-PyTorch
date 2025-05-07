import torch

from semanticsegmentor.ml_models import UNet, UNetTunable


def test_unet_model():
    in_channels = 1
    out_classes = 1
    unet = UNet(in_channels=in_channels,
                out_classes=out_classes,
                bilinear=False)

    in_tensor_1 = torch.FloatTensor(2, 1, 80, 250)
    in_tensor_2 = torch.FloatTensor(4, 1, 100, 100)
    in_tensor_3 = torch.FloatTensor(6, 1, 264, 264)

    out_tensor_1 = unet(in_tensor_1)
    out_tensor_2 = unet(in_tensor_2)
    out_tensor_3 = unet(in_tensor_3)

    assert out_tensor_1.shape == (2, 1, 80, 250)
    assert out_tensor_2.shape == (4, 1, 100, 100)
    assert out_tensor_3.shape == (6, 1, 264, 264)


def test_unet_tunable_model():
    in_channels = 1
    out_classes = 1
    unet_tune1 = UNetTunable(in_channels=in_channels,
                             out_classes=out_classes,
                             depth=4, filters=3,
                             dilation=1, dropout=False,
                             up_mode='upconv')

    unet_tune2 = UNetTunable(in_channels=in_channels,
                             out_classes=out_classes,
                             depth=4, filters=3,
                             dilation=2, dropout=True,
                             up_mode='upsample')

    unet_tune3 = UNetTunable(in_channels=in_channels,
                             out_classes=out_classes,
                             depth=4, filters=3, batch_norm=False,
                             dilation=2, dropout=True,
                             up_mode='upsample',
                             attention=True)

    in_tensor_1 = torch.FloatTensor(2, 1, 80, 250)
    in_tensor_2 = torch.FloatTensor(4, 1, 100, 100)
    in_tensor_3 = torch.FloatTensor(6, 1, 264, 264)

    out_tensor_1 = unet_tune1(in_tensor_1)
    out_tensor_11 = unet_tune2(in_tensor_1)
    out_tensor_111 = unet_tune3(in_tensor_1)

    out_tensor_2 = unet_tune1(in_tensor_2)
    out_tensor_22 = unet_tune2(in_tensor_2)
    out_tensor_222 = unet_tune3(in_tensor_2)

    out_tensor_3 = unet_tune1(in_tensor_3)
    out_tensor_33 = unet_tune2(in_tensor_3)
    out_tensor_333 = unet_tune3(in_tensor_3)

    assert out_tensor_1.shape == (2, 1, 80, 250)
    assert out_tensor_11.shape == (2, 1, 80, 250)
    assert out_tensor_111.shape == (2, 1, 80, 250)
    assert out_tensor_2.shape == (4, 1, 100, 100)
    assert out_tensor_22.shape == (4, 1, 100, 100)
    assert out_tensor_222.shape == (4, 1, 100, 100)
    assert out_tensor_3.shape == (6, 1, 264, 264)
    assert out_tensor_33.shape == (6, 1, 264, 264)
    assert out_tensor_333.shape == (6, 1, 264, 264)


# from unet.utils import summary
#
# unet_tune3 = UNetTunable(in_channels=1,
#                          out_classes=1,
#                          depth=4, filters=3, batch_norm=False,
#                          dilation=1, dropout=True,
#                          up_mode='upsample',
#                          attention=False)
#
# result, params_info = summary(unet_tune3.cuda(), tuple((1, 80, 256)))
# print(f"Total parameters in the model:{params_info[0]}")
# print(f"Trainable parameters in the model:{params_info[1]}")
# print(result)
