from torchvision import transforms, datasets

def cifar_dataset():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10(
        root='./',
        train=True,
        transform=train_transform,
        download=True
        )

    test_data = datasets.CIFAR10(
        root='./',
        train=False,
        transform=test_transform,
        download=True
    )
    return train_data, test_data

if __name__=='__main__':
    train_data, test_data = cifar_dataset()

    image, label = train_data[0]
    print(f"image size: {image.size()}")
    print(f'label: {label}')