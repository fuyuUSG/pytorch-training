from torchvision import transforms, datasets

def cifar_dataset():
    # 正規化(平均0.5、標準偏差0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10(
        root='./',
        train=True,
        transform=transform,
        download=True
        )

    test_data = datasets.CIFAR10(
        root='./',
        train=False,
        transform=transform,
        download=True
    )
    return train_data, test_data

if __name__=='__main__':
    train_data, test_data = cifar_dataset()

    image, label = train_data[0]
    print(f"image size: {image.size()}")
    print(f'label: {label}')