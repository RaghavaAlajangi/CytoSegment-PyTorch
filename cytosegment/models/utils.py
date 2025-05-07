from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def add_params_to_jit_model(model_jit_path, params_dict):
    model = torch.jit.load(model_jit_path, map_location="cpu")
    seq_scripted = torch.jit.script(model)
    extra_files = {"meta": str(params_dict)}
    torch.jit.save(seq_scripted, model_jit_path, _extra_files=extra_files)


def convert_torch_to_onnx(torch_ckp_path, img_size):
    cuda_device = torch.device("cpu")
    ckp = torch.load(torch_ckp_path, map_location=cuda_device)
    model = ckp["model_instance"]
    model.load_state_dict(ckp["model_state_dict"])
    model.eval()

    onnx_path = Path(torch_ckp_path).with_suffix(".onnx")

    batch_size = 8
    dummy_input = torch.randn(
        batch_size, 1, img_size[0], img_size[1], requires_grad=True
    )
    dummy_input = dummy_input.to(cuda_device, dtype=torch.float32)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        # model input (or a tuple for multiple inputs)
        dummy_input,
        # where to save the model (can be a file-like object)
        str(onnx_path),
        # store the trained  weights inside the model file
        export_params=True,
        # the ONNX version to export the model to
        opset_version=11,
        # whether to execute constant folding for optimization
        do_constant_folding=True,
        # the model's input names
        input_names=["input"],
        # the model's output names
        output_names=["output"],
        # variable length axes
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def init_weights(net, init_type="HeNormal", gain=0.02):
    """
    Initializes the weights of a network.
    Parameters
    ----------
    net
        Pass the network to be initialized
    init_type
        Specify the type of initialization to be used
    gain
        Scale the weights of the network
    Returns
    -------
    The initialized network
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "HeNormal":
                init.kaiming_normal_(
                    m.weight.data, mode="fan_in", nonlinearity="relu"
                )
            elif init_type == "HeUniform":
                init.kaiming_uniform_(
                    m.weight.data, mode="fan_in", nonlinearity="relu"
                )
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "init method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"Initialize network with {init_type}")
    return net.apply(init_func)


def summary(
    model, input_size, batch_size=-1, device=torch.device("cpu"), dtypes=None
):
    # Bring model to CPU device
    model = model.cpu()
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ""

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(
                    torch.LongTensor(list(module.weight.size()))
                )
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(
                    torch.LongTensor(list(module.bias.size()))
                )
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(
            module, nn.ModuleList
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [
        torch.rand(2, *in_size).type(dtype).to(device=device)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "-" * 64 + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #"
    )
    summary_str += line_new + "\n"
    summary_str += "=" * 64 + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(
        np.prod(sum(input_size, ())) * batch_size * 4.0 / (1024**2.0)
    )
    total_out_size = abs(2.0 * total_output * 4.0 / (1024**2.0))
    total_params_size = abs(total_params * 4.0 / (1024**2.0))
    total_size = total_params_size + total_out_size + total_input_size
    non_train_params = total_params - trainable_params

    summary_str += f"{'=' * 64}\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {trainable_params:,}\n"
    summary_str += f"Non-trainable params: {non_train_params:,}\n"
    summary_str += f"{'-' * 64}\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += f"Forward/backward pass size(MB): {total_out_size:0.2f}\n"
    summary_str += f"Params size (MB): {total_params_size:0.2f}\n"
    summary_str += f"Estimated Total Size (MB): {total_size:0.2f}\n"
    summary_str += f"{'-' * 64}\n"
    # return summary
    return summary_str, (total_params, trainable_params)
