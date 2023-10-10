from pickletools import optimize
from matplotlib.pyplot import axis
import torch
from dataset import MyDataset
from k_fold import get_k_fold_data
from unet_model import UNet
from torch import optim
from Dice import SoftDiceLoss


def train(i,train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l=[]
    test_acc_max = 0
    for epoch in range(num_epochs):  #迭代40次
        batch_count = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += torch.sum((torch.argmax(y_hat.data,1))==y.data).float().item()/(224*224)
            n += y.shape[0]
            batch_count += 1
    #至此，每个epoches完成
        test_acc_sum= evaluate_accuracy(test_iter, net)
        train_l_min_l.append(train_l_sum/batch_count)
        train_acc_max_l.append(train_acc_sum/n)
        test_acc_max_l.append(test_acc_sum)

        print('fold %d epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (i+1,epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc_sum))
        ###保存
        if test_acc_max_l[-1] > test_acc_max:
            test_acc_max = test_acc_max_l[-1]
            torch.save(net.state_dict(), "./results/K{:}_model_best.pth".format(i+1))
            print("saving K{:}_model_best.pth ".format(i))
    ####选择测试准确率最高的那一个epoch对应的数据，打印并写入文件
    index_max=test_acc_max_l.index(max(test_acc_max_l))
    f = open("./results.txt", "a")
    if i==0:
        f.write("fold"+"   "+"train_loss"+"       "+"train_acc"+"      "+"test_acc")
    f.write('\n' +"fold"+str(i+1)+":"+str(train_l_min_l[index_max]) + " ;" + str(train_acc_max_l[index_max]) + " ;" + str(test_acc_max_l[index_max]))
    f.close()
    print('fold %d, train_loss_min %.4f, train acc max%.4f, test acc max %.4f'
            % (i + 1, train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max]))
    return train_l_min_l[index_max],train_acc_max_l[index_max],test_acc_max_l[index_max]


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()/(224*224)
                net.train() # 改回训练模式
            else:
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()/(224*224)
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()/(224*224)
            n += y.shape[0]
    return acc_sum / n

def k_fold(k,image_dir,num_epochs,device,batch_size):
    train_k = './train_k.txt'
    test_k = './test_k.txt'
    #loss_acc_sum,train_acc_sum, test_acc_sum = 0,0,0
    Ktrain_min_l = []
    Ktrain_acc_max_l = []
    Ktest_acc_max_l = []
    for i in range(k):
        net=UNet(n_channels=3,n_classes=1)
        optimizer=optim.RMSprop(net.parameters(), lr=0.01, weight_decay=1e-8, momentum=0.9)
        loss = torch.nn.BCEWithLogitsLoss()
        get_k_fold_data(k, i, image_dir)

        train_data = MyDataset(is_train=True, root=train_k)
        test_data = MyDataset(is_train=False, root=test_k)

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=1)
        # 修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        loss_min,train_acc_max,test_acc_max=train(i,train_loader,test_loader, net, loss, optimizer, device, num_epochs)

        Ktrain_min_l.append(loss_min)
        Ktrain_acc_max_l.append(train_acc_max)
        Ktest_acc_max_l.append(test_acc_max)
    return sum(Ktrain_min_l)/len(Ktrain_min_l),sum(Ktrain_acc_max_l)/len(Ktrain_acc_max_l),sum(Ktest_acc_max_l)/len(Ktest_acc_max_l)

if __name__ == '__main__':
    batch_size=1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k=5
    image_dir='./shuffle_data.txt'
    num_epochs=40
    loss_k,train_k, valid_k=k_fold(k,image_dir,num_epochs,device,batch_size)
    f=open("./results.txt","a")
    f.write('\n'+"avg in k fold:"+"\n"+str(loss_k)+" ;"+str(train_k)+" ;"+str(valid_k))
    f.close()
    print('%d-fold validation: min loss rmse %.5f, max train rmse %.5f,max test rmse %.5f' % (k,loss_k,train_k, valid_k))
