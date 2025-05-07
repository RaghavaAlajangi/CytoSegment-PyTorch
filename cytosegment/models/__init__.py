from .tune_unet import TunableUNet
from .bench_unet import BenchmarkUNet
from .small_unet import UNet
from .utils import (add_params_to_jit_model, convert_torch_to_onnx,
                    init_weights, summary)


def get_model_with_params(params):
    assert {"model"}.issubset(params)
    model_params = params.get("model")
    assert {"type"}.issubset(model_params)
    model_type = model_params.get("type")
    assert {"weight_init"}.issubset(model_params)
    weight_init = model_params.get("weight_init")

    if model_type.lower() == "unet":
        assert {"in_channels", "out_classes"}.issubset(model_params)
        model = UNet(in_channels=model_params.get("in_channels"),
                     out_classes=model_params.get("out_classes"))

        if weight_init.lower() == "default":
            return model
        else:
            return init_weights(model, init_type=weight_init)

    if model_type.lower() == "benchunet":
        assert {"in_channels", "out_classes"}.issubset(model_params)
        assert {"bilinear"}.issubset(model_params)
        model = BenchmarkUNet(in_channels=model_params.get("in_channels"),
                              out_classes=model_params.get("out_classes"),
                              bilinear=model_params.get("bilinear"))

        if weight_init.lower() == "default":
            return model
        else:
            return init_weights(model, init_type=weight_init)

    if model_type.lower() == "tunableunet":
        assert {"in_channels", "out_classes"}.issubset(model_params)
        assert {"conv_block"}.issubset(model_params)
        assert {"depth", "filters"}.issubset(model_params)
        assert {"batch_norm", "dropout"}.issubset(model_params)
        assert {"dilation", "relu"}.issubset(model_params)
        assert {"up_mode", "attention"}.issubset(model_params)
        model = TunableUNet(in_channels=model_params.get("in_channels"),
                            out_classes=model_params.get("out_classes"),
                            conv_block=model_params.get("conv_block"),
                            depth=model_params.get("depth"),
                            filters=model_params.get("filters"),
                            dilation=model_params.get("dilation"),
                            dropout=model_params.get("dropout"),
                            batch_norm=model_params.get("batch_norm"),
                            up_mode=model_params.get("up_mode"),
                            attention=model_params.get("attention"),
                            relu=model_params.get("relu"))
        if weight_init.lower() == "default":
            print("Default initialization of network is used!")
            return model
        else:
            return init_weights(model, init_type=weight_init)
