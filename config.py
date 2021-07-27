class DefaultConfig(object):
    train_root = 'D:/AProgram/data/4x_div2k_file.h5'
    validation_root = 'D:/AProgram/data/4x_div2k_file_valid.h5'
    lr = 5e-4
    batch_size = 32
    num_workers = 8
    epoch = 200

    cuda = True

    opts1 = {
        'title': 'train_loss',
        'xlabel': 'epoch',
        'ylabel': 'loss',
        'width': 300,
        'height': 300,
    }

    opts2 = {
        'title': 'eval_psnr',
        'xlabel': 'epoch',
        'ylabel': 'psnr',
        'width': 300,
        'height': 300,
    }

opt = DefaultConfig()