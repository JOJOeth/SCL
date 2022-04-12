import torchvision


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "regnet_y_32gf": torchvision.models.regnet_y_32gf(pretrained=pretrained),
        "regnet_y_16gf": torchvision.models.regnet_y_16gf(pretrained=pretrained),
        "regnet_y_400mf": torchvision.models.regnet_y_400mf(pretrained=pretrained),
        "regnet_y_800mf": torchvision.models.regnet_y_800mf(pretrained=pretrained),
        "regnet_y_1_6gf": torchvision.models.regnet_y_800mf(pretrained=pretrained)
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
