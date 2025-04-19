from dataset import MyDataset
from SignalProcessing import *
from torchvision.utils import make_grid, save_image
import tqdm
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn
import torch
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
from module.Dense import *
from module.JCT_Hardanger import *
from module.DFCN_Hardanger import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_rate", type=float, default=0.98, help="lr decay rate")
parser.add_argument('--lambda1', default='0.2', type=float, help='hyper paramater') 
parser.add_argument('--lossfunction_type', default='1', type=str, help='loss_type') 
parser.add_argument('--save_path', default=r'Type your save path', type=str, help='save path')
parser.add_argument('--model', default='DFCN', type=str, help='model') #
parser.add_argument('--retain_rato', default='0.75', type=str, help='M  parameter== 1-miss_ratio') #
args = parser.parse_args()

"""
    lossfunction_type == 1 denotes the Regeneation loss function

    lossfunction_type == 2 denotes the Original Frobenius function
"""
   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator= DFCN_Hardanger()
generator.to(device)

train_path_x = "Type your input path"
train_path_y = "Type your label path"

full_dataset = MyDataset(train_path_x, train_path_y)
test_size = int(len(full_dataset) * 0.2)
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset,[train_size, test_size])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

optim_gen = optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1,args.b2))
generator.apply(init_weights)
writer = SummaryWriter()
writer_dict = {'writer':writer}
writer_dict["train_global_steps"]=0
temp_sets1 = []   
temp_sets2 = []   
temp_sets3 = []   
temp_sets4 = []  
start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  


for epoch in range(args.epochs):

    lr = args.lr_gen * (args.decay_rate ** epoch)
    for param_group in optim_gen.param_groups:
        param_group['lr'] = lr
    D_train_loss = 0.0
    G_train_loss = 0.0
    gen_step = 0

    """
    训练网络
    """
    generator = generator.train()
    
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0): 

        global_steps = writer_dict['train_global_steps']
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)

        """
            lossfunction_type == 1 denotes the Regeneation loss function

            lossfunction_type == 2 denotes the  Original L2 function
        """

        # ---------
        #  Train 
        # ---------

        mask = torch.ones_like(batch_x, dtype=torch.float32).to(device=device)
        mask[batch_x == 0] = 0      
        M = mask_sampling(mask, args.model, args.retain_rato).float().to(device)
    
        x_recon = generator(batch_x)
        x_recon_l2_loss = (x_recon - batch_y).norm(2)
        x_rafix = generator(torch.mul(M, x_recon))
        x_regen_l2_loss = torch.mul(1-M, x_recon-x_rafix).norm(2)
        x_stable_l2_loss = torch.mul(1-mask, batch_y-x_rafix).norm(2)
        x_rafix_l2_loss = (x_regen_l2_loss + x_stable_l2_loss)

        if args.lossfunction_type == '1':
            G_loss = (x_recon_l2_loss + args.lambda1 * x_rafix_l2_loss)/batch_x.shape[0]
        else:
            G_loss = x_recon_l2_loss/batch_x.shape[0]

        G_train_loss += G_loss.item()
        optim_gen.zero_grad()    
        G_loss.backward()
        optim_gen.step()
        gen_step += 1

        if gen_step and batch_idx1 % 100 == 0:
            sample_imgs = x_recon[:25]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            gen_datas_array = sample_imgs.detach().cpu().numpy()
            directory = args.save_path + r'\save_gendata'   # 目录路径
            filename = f'./generated_datas_epoch{epoch}_batch{batch_idx1}.npz'  # 指定文件名
            filepath = directory + '/' + filename
            np.savez(filepath, gen_datas_array )    
  
    G_train_loss = G_train_loss / (batch_idx1+1)  # 本次epoch的平均训练G_train_loss

    """
         valid
    """  
    generator = generator.eval()
    error_set = 0.0
    RMSE_set = 0.0
    R_square_set = 0.0
    G_val_loss = 0.0

    for batch_idx2, (val_x, val_y) in enumerate(test_loader, 0):
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)

        with torch.no_grad():

            mask2 = torch.ones_like(val_x, dtype=torch.float32).to(device=device)
            mask2[val_x == 0] = 0      
            M2 = mask_sampling(mask2, args.model, args.retain_rato).float().to(device)

            x_recon2 = generator(val_x)
            x_recon_l2_loss2 = (val_y - x_recon2).norm(2)
            x_rafix2 = generator(torch.mul(M2, x_recon2))
            x_regen_l2_loss2 = torch.mul(1-M2, x_recon2-x_rafix2).norm(2)
            x_stable_l2_loss2 = torch.mul(1-mask2, val_y-x_rafix2).norm(2)               
            x_rafix_l2_loss2 = (x_regen_l2_loss2 + x_stable_l2_loss2)
            if args.lossfunction_type == '1':
                G_loss2 = (x_recon_l2_loss2 + args.lambda1 * x_rafix_l2_loss2)/val_x.shape[0]   
            else: 
                G_loss2 = x_recon_l2_loss2/val_x.shape[0]  
            error = calculate_error(x_recon2 , val_y)  
            RMSE = calculate_rmse(x_recon2 , val_y)
            G_val_loss += G_loss2.item()  
        
        error_set += error  
        RMSE_set += RMSE
       
    error_set = error_set / (batch_idx2 + 1)
    RMSE_set = RMSE_set / (batch_idx2 + 1)
    G_val_loss = G_val_loss / (batch_idx2+1)
    loss_set = [G_train_loss, G_val_loss]
    temp_sets1.append(loss_set)
    temp_sets2.append(error_set)
    temp_sets3.append(RMSE_set)
    temp_sets4.append(R_square_set)
    
    print(
            "[Epoch %d/%d] [G_train_loss: %8f] [G_val_loss: %8f] [ERROR: %8f] [RMSE: %8f]"
            % (epoch, args.epochs, G_train_loss, G_val_loss, error_set, RMSE_set)
        )

    model_name = f'model_epoch{epoch+1}' 
    torch.save(generator, os.path.join(args.save_path + r'\save_model', model_name+'.pth')) 

end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  

np.savetxt(args.save_path + r'\save_model\\loss_sets.txt', temp_sets1, fmt='%.8f')
np.savetxt(args.save_path + r'\save_model\\error_sets.txt', temp_sets2, fmt='%.8f')
np.savetxt(args.save_path + r'\save_model\\RMSE_sets.txt', temp_sets3, fmt='%.8f')

