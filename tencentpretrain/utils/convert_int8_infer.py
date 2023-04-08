import bitsandbytes as bnb
import torch

def convert_linear_to_bnb(float_linear):
    new_layer = bnb.nn.Linear8bitLt(
        float_linear.in_features,
        float_linear.out_features,
        bias=float_linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0
    )
    new_layer._parameters["weight"] = bnb.nn.Int8Params(
        float_linear.weight.data.cpu(),
        requires_grad=False,
        has_fp16_weights=False
    )
    if float_linear.bias is not None:
        new_layer._parameters["bias"] = float_linear.bias
    return new_layer


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


def transfer_linear_to_int8linear(model):
    linear_layers = {
        k: v for k, v in model.named_modules() if isinstance(v, torch.nn.Linear)
    }

    for name, layer in linear_layers.items():
        new_layer = convert_linear_to_bnb(layer)
        set_layer(model, name, new_layer)
    model.cuda()