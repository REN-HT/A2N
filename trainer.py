from model.AAN import AAN
from dataset.DataSet import DataSet, ValidDataset
from config import opt
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
import visdom
from torch.autograd import Variable
from tool import calculate_psnr, convert_rgb_to_y, denormalize
import copy
from model.AAN import L1_Charbonnier_loss


# 利用预训练的模型来更新网络参数，需要保证参数名匹配！！！否则无法完成更新
def transfer_model(pretrain_file, model):
    pretrain_dict = torch.load(pretrain_file)
    model_dict = model.state_dict()
    pretrain_dict = transfer_state_dict(pretrain_dict, model_dict)
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrain_dict, model_dict):
    state_dict = {}
    count = 0
    for k, v in pretrain_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
            count += 1
    if count == 0:
        print('no parameters update!!!')
    else:
        print('update successfully!!!')
    return state_dict


def train():
    # 数据载入
    train_data = DataSet(opt.train_root)
    valid_data = ValidDataset(opt.validation_root)

    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_data_loader = DataLoader(valid_data, batch_size=1)

    # 定义网络，优化器，损失
    net = AAN()
    # net = transfer_model('best_2xAAN_weight.pth', net)
    # static_dic = torch.load('best_2xAAN_weight.pth')
    # net.load_state_dict(static_dic)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    # criterion = nn.L1Loss()
    criterion = L1_Charbonnier_loss()
    if opt.cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    best_weight = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_psnr = 0

    vis = visdom.Visdom(env=u'aan')
    for epoch in range(opt.epoch):
        train_loss = 0
        net.train()
        for ii, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            lr, hr = Variable(data[0]), Variable(data[1])
            if opt.cuda:
                lr = lr.cuda()
                hr = hr.cuda()
            # 梯度清零
            sr = net(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('epoch:{},loss:{:.3f}'.format(epoch+1, train_loss/(ii+1)))
        vis.line(X=torch.Tensor([epoch+1]), Y=torch.Tensor([train_loss/len(train_data_loader)]), win='win1', update='append', opts=opt.opts1)
        if (epoch+1) % 5 == 0:
            torch.save(net.state_dict(), '4xAAN_weight{}.pth'.format((epoch+1) / 5))

        # 验证
        net.eval()
        res = 0
        for item in valid_data_loader:
            lr, hr = item
            if opt.cuda:
                lr = lr.cuda()
                hr = hr.cuda()
            with torch.no_grad():
                sr = net(lr)
            sr = convert_rgb_to_y(denormalize(sr.squeeze(0)))
            hr = convert_rgb_to_y(denormalize(hr.squeeze(0)))
            res += calculate_psnr(sr, hr)
        avg_psnr = res/len(valid_data_loader)
        print('eval_psnr:{:.3f}'.format(avg_psnr))
        vis.line(X=torch.Tensor([epoch+1]), Y=torch.Tensor([avg_psnr]), win='win2', update='append', opts=opt.opts2)
        # 记录最优迭代
        if best_psnr < avg_psnr:
            best_epoch = epoch+1
            best_psnr = avg_psnr
            best_weight = copy.deepcopy(net.state_dict())
    print('best_epoch {}, best_psnr {:.3f}'.format(best_epoch, best_psnr))
    torch.save(best_weight, 'best_4xAAN_weight.pth')


if __name__ == '__main__':
    train()
