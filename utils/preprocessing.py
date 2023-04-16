import albumentations as A


"""
Preprocessing pipeline for the images.
"""
ssn_preprocessing_pipeline = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.Flip(p=0.5),
        A.CropAndPad(percent=(0.2, 0.2), p=0.5),
    ]),
    A.OneOf([
        A.Compose([
            A.Resize(112, 112),
            A.Resize(224, 224),
        ]),
        A.GaussianBlur(),
        A.MotionBlur(),
    ]),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.GaussNoise(var_limit=(0.05, 0.1), always_apply=True)
    ]),
    A.OneOf([
        A.PixelDropout(),
        A.CoarseDropout(),
    ])
])