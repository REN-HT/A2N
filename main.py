import PIL.Image
from PIL import Image
from torchvision import transforms as T
from model.AAN import AAN
import torch
import os
import numpy as np
from tool import denormalize, convert_rgb_to_y, calculate_psnr
from config import opt
import cv2
import h5py


def resize(img, h, w):
    trans = T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC)
    return trans(img)


def real_test(path):
    img = Image.open(path).convert('RGB')
    # w, h = img.size
    # y, b, r = img.split()
    # b = resize(b, h*4, w*4)
    # r = resize(r, h*4, w*4)
    input = T.ToTensor()(img).unsqueeze(0)
    model = AAN()
    model_state_dic = torch.load('D:/AProgram/SR/AAN/best_4xAAN_weight.pth')
    model.load_state_dict(model_state_dic)
    if opt.cuda:
        input = input.cuda()
        model = model.cuda()
    with torch.no_grad():
        out4x = model(input).clamp(0.0, 1.0)
    out4x = out4x.squeeze(0)
    out4x = T.ToPILImage()(out4x)
    out4x.show()
    # imgout = Image.merge('YCbCr', (out4x, b, r))


def test(path, scale):
    gt = Image.open(path).convert('RGB')
    w, h = gt.size
    w, h = (w // scale) * scale, (h // scale) * scale
    img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
    lr = img.resize((w // scale, h // scale), resample=PIL.Image.BICUBIC)
    input = T.ToTensor()(lr).unsqueeze(0)
    model = AAN()
    model_state_dic = torch.load('D:/AProgram/SR/AAN/best_4xAAN_weight.pth')
    model.load_state_dict(model_state_dic)
    if opt.cuda:
        input = input.cuda()
        model = model.cuda()
    with torch.no_grad():
        out = model(input).clamp(0.0, 1.0)
    out = out.squeeze(0)
    out = T.ToPILImage()(out)
    lr.show()
    out.show()


# 计算峰值信噪比
def PSNRRGB(root, scale):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net = AAN()
    model_state_dic = torch.load('D:/AProgram/SR/AAN/4xAAN_weight38.0.pth')
    net.load_state_dict(model_state_dic)
    res = 0
    for path in img_paths:
        gt = Image.open(path).convert('RGB')
        w, h = gt.size
        w, h = (w // scale) * scale, (h // scale) * scale
        img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
        lr = img.resize((w//scale, h//scale), resample=PIL.Image.BICUBIC)
        lr = to_tensor(lr).unsqueeze(0)
        if opt.cuda:
            lr = lr.cuda()
            net = net.cuda()
        with torch.no_grad():
            preds = net(lr).squeeze(0)
        labels = to_tensor(img)
        preds = convert_rgb_to_y(denormalize(preds.cpu()))
        labels = convert_rgb_to_y(denormalize(labels))
        # 裁剪边缘部分
        preds = preds[scale:-scale, scale:-scale]
        labels = labels[scale:-scale, scale:-scale]

        res += calculate_psnr(preds, labels)
    print('PSNR:{:.3f}'.format(res/len(img_paths)))


def SSIM(root, scale):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net = AAN()
    model_state_dic = torch.load('D:/AProgram/SR/AAN/best_2xAAN_weight.pth')
    net.load_state_dict(model_state_dic)
    res = 0
    for path in img_paths:
        gt = Image.open(path).convert('RGB')
        w, h = gt.size
        w, h = (w // scale) * scale, (h // scale) * scale
        img = gt.resize((w, h), resample=PIL.Image.BICUBIC)
        lr = img.resize((w//scale, h//scale), resample=PIL.Image.BICUBIC)
        input = to_tensor(lr).unsqueeze(0)
        if opt.cuda:
            input = input.cuda()
            net = net.cuda()
        with torch.no_grad():
            preds = net(input).squeeze(0)
        labels = to_tensor(img)

        preds = convert_rgb_to_y(denormalize(preds.cpu()))
        labels = convert_rgb_to_y(denormalize(labels))

        preds = preds.numpy()
        labels = labels.numpy()

        res += calculate_ssim(preds, labels)

    print('SSIM:{:.4f}'.format(res / len(img_paths)))


# 计算两幅图片结构相似比
def calculate_ssim(img1, img2):

    # 固定系数,1.0为最大像数值，由于这里为Tensor类型,所以最大为1，如果像素值范围是【0-255】则为255
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# 创建h5_file文件
def create_h5_file(root, scale):
    h5_file = h5py.File('D:/AProgram/data/4x_div2k_file.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    index = 0

    for img_path in img_paths:
        hr = Image.open(img_path).convert('RGB')
        lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)
        hr = T.ToTensor()(hr)
        lr = T.ToTensor()(lr)
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)
        index += 1

    h5_file.close()


def create_h5_file_valid(root, scale):
    h5_file = h5py.File('D:/AProgram/data/4x_div2k_file_valid.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    valid_img_names = os.listdir(root)
    paths = [os.path.join(root, name) for name in valid_img_names]
    pos = 0

    for path in paths:
        hr = Image.open(path).convert('RGB')
        for hr in T.FiveCrop(size=(hr.height//2, hr.width//2))(hr):
            hr = hr.resize(((hr.width//scale)*scale, (hr.height//scale)*scale), resample=PIL.Image.BICUBIC)
            lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)

            hr = T.ToTensor()(hr)
            lr = T.ToTensor()(lr)

            lr_group.create_dataset(str(pos), data=lr)
            hr_group.create_dataset(str(pos), data=hr)
            pos += 1

    h5_file.close()


if __name__=='__main__':
    path = 'F:/dataset/SuperResolutionDataset/Test/Mix/butterfly_GT.bmp'
    path1 = 'F:/dataset/SuperResolutionDataset/Test/Set5'
    path2 = 'F:/dataset/SuperResolutionDataset/480x480DIV2K_train_HR'
    path3 = 'F:/dataset/SuperResolutionDataset/valid/temp'
    test(path, 4)
    # PSNRRGB(path1, 4)
    # SSIM(path1, 2)
    # create_h5_file(path2, 4)
    # create_h5_file_valid(path3, 4)


