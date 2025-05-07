from torch.optim import SGD, Adam


def get_optimizer_with_params(params, model):
    assert {"optimizer"}.issubset(params)
    optimizer_params = params.get("optimizer")
    assert {"type", "learn_rate"}.issubset(optimizer_params)
    optimizer_type = optimizer_params.get("type")
    learn_rate = optimizer_params.get("learn_rate")

    if optimizer_type.lower() == "adam":
        return Adam(model.parameters(), lr=learn_rate, eps=1e-07)

    if optimizer_type.lower() == "sgd":
        assert {"momentum"}.issubset(optimizer_params)
        momentum = optimizer_params.get("momentum")
        return SGD(model.parameters(), lr=learn_rate, momentum=momentum)
