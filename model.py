import torch
from torch import nn
from torch.nn.init import kaiming_uniform
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

class RelationNetworks(nn.Module):
    def __init__(self, n_vocab, conv_hidden=24, embed_hidden=32,
                 lstm_hidden=128, mlp_hidden=256, classes=29):
        super(RelationNetworks, self).__init__()

        COORS8 = np.zeros((8, 8, 2))
        for i in range(8):
            for j in range(8):
                COORS8[i, j, 0] = (i-3.5)/3.5
                COORS8[i, j, 1] = (j-3.5)/3.5

        self.ma_conv1 = nn.Conv2d(1, 24, 5, stride=2, padding=2)
        self.ma_batchNorm1 = nn.BatchNorm2d(24)
        self.ma_conv2 = nn.Conv2d(24, 32, 5, stride=2, padding=2)
        self.ma_batchNorm2 = nn.BatchNorm2d(32)
        self.ma_conv3 = nn.Conv2d(32, 16, 8, stride=1, padding=0)
        self.ma_batchNorm3 = nn.BatchNorm2d(16)
        
        self.seg_conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.seg_batchNorm1 = nn.BatchNorm2d(16)
        self.seg_conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.seg_batchNorm2 = nn.BatchNorm2d(16)
        self.seg_conv3 = nn.Conv2d(16, 16, 8, stride=1, padding=0)
        self.seg_batchNorm3 = nn.BatchNorm2d(16)
#        self.seg_conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
#        self.seg_batchNorm3 = nn.BatchNorm2d(24)
        
        self.g2_fc1 = nn.Linear(192, 256)
#        self.g2_fc2 = nn.Linear(256, 256)
#        self.g2_fc3 = nn.Linear(256, 256)
        self.g2_fc4 = nn.Linear(256, 256)
        
        self.g3_fc1 = nn.Linear(224, 256)
#        self.g3_fc2 = nn.Linear(256, 256)
#        self.g3_fc3 = nn.Linear(256, 256)
        self.g3_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(512, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, 29)
        

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.n_concat = conv_hidden * 2 + lstm_hidden + 4

        self.conv_hidden = conv_hidden
        self.lstm_hidden = lstm_hidden
        self.mlp_hidden = mlp_hidden
        
        coors_tensor = torch.FloatTensor(COORS8)
        
        if torch.cuda.is_available():
            self.coors_tensor = Variable(coors_tensor).cuda()
        else:
            self.coors_tensor = Variable(coors_tensor)
        #self.initialize_weights()

    def forward(self, apps, masks, num_layers, question, question_len):
        
        batch_size, MX_N, _, _ = masks.size()
        
        question_len = question_len.data.cpu().numpy().tolist()
        
        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                    batch_first=True)
        _, (h, c) = self.lstm(embed_pack)
        
        qst = h.permute(1, 0, 2) # batchsize*self.lstm_hidden

        num_layers_value = num_layers.data.cpu().numpy()
        
        """masks"""
        masks = masks.unsqueeze(2).view(batch_size*MX_N, 1, 32, 32)
        x_m = self.ma_conv1(masks)
        x_m = F.relu(x_m)
        x_m = self.ma_batchNorm1(x_m)
        x_m = self.ma_conv2(x_m)
        x_m = F.relu(x_m)
        x_m = self.ma_batchNorm2(x_m)
        x_m = self.ma_conv3(x_m)
        x_m = F.relu(x_m)
        x_m = self.ma_batchNorm3(x_m)
        x_m = x_m.view(batch_size, MX_N, 16)
        # imshow
#        if not torch.cuda.is_available():
#            import matplotlib.pyplot as plt
#            for b in range(batch_size):
#                img = image[b].permute(1, 2, 0)
#                img = img.data.cpu().numpy()
#                img_tmp = np.zeros_like(img)
#                img_tmp[:,:,0] = img[:,:,2]
#                img_tmp[:,:,2] = img[:,:,0]
##                print (img_tmp)
#                
#    #            print (img_tmp)
#                fig = plt.figure()
#                ax = fig.add_subplot(111)
#                ax.imshow(img)
#                plt.show()
#                for l in range(num_layers_value[b]):
#                    img = apps[b, l].permute(1, 2, 0)
#                    img = img.data.cpu().numpy()
#                    img_tmp = np.zeros_like(img)
#                    img_tmp[:,:,0] = img[:,:,2]
#                    img_tmp[:,:,2] = img[:,:,0]
#                    fig = plt.figure()
#                    ax = fig.add_subplot(111)
#                    ax.imshow(img)
#                    plt.show()
                
        """segs"""
        x_all = apps.view(batch_size*MX_N, 3, 32, 32)
        x_all = self.seg_conv1(x_all) # num_layers*128*128*3
        x_all = F.relu(x_all)
        x_all = self.seg_batchNorm1(x_all)
        x_all = F.max_pool2d(x_all, 2)
        x_all = self.seg_conv2(x_all)
        x_all = F.relu(x_all)
        x_all = self.seg_batchNorm2(x_all)
        x_all = F.max_pool2d(x_all, 2)
        x_all = self.seg_conv3(x_all)
        x_all = F.relu(x_all)
        x_all = self.seg_batchNorm3(x_all)
        
#        print (x_all.size())
#        x_all = x_all.contiguous().view(batch_size, MX_N, 24, 8, 8)
#        x_flat = x_all.permute(0, 1, 3, 4, 2) # batch_size*N*8*8*24
#        x_flat = x_flat.contiguous().view(-1, 8*8*24) # batch_size*N*8*8, 24
##            print (x_flat) # N, 1536
#        x_all = self.seg_fc(x_flat) # batch_size*N, 22
        x_all = x_all.view(batch_size, MX_N, 16)
        x_all = torch.cat([x_all, x_m], -1) # batch_size, MX_N, 32
#        x_all = torch.cat([x_all, coors, sizes.unsqueeze(-1).repeat(1, 1, 2)], -1)
        x_i = x_all.unsqueeze(1).repeat(1, MX_N, 1, 1)
        x_j = x_all.unsqueeze(2).repeat(1, 1, MX_N, 1)
        
        x_i_3 = x_all.unsqueeze(1).unsqueeze(1).repeat(1, MX_N, MX_N, 1, 1)
        x_j_3 = x_all.unsqueeze(2).unsqueeze(1).repeat(1, MX_N, 1, MX_N, 1)
        x_k_3 = x_all.unsqueeze(2).unsqueeze(2).repeat(1, 1, MX_N, MX_N, 1)
        
        qst_2 = qst.unsqueeze(1).repeat(1, MX_N, MX_N, 1)
        qst_3 = qst.unsqueeze(1).unsqueeze(1).repeat(1, MX_N, MX_N, MX_N, 1)
#        print (qst_2)
#        print (qst_3)
        concat_vec_2 = torch.cat([x_i, x_j, qst_2], -1)
        concat_vec_3 = torch.cat([x_i_3, x_j_3, x_k_3, qst_3], -1)
#        print (concat_vec_2.size())
#        print (concat_vec_3.size())
        
        concat_vec_2 = concat_vec_2.view(-1, 192)
        concat_vec_3 = concat_vec_3.view(-1, 224)
        
        """g2 """
        x_2 = self.g2_fc1(concat_vec_2)
        x_2 = F.relu(x_2)
#        x_2 = self.g2_fc2(x_2)
#        x_2 = F.dropout(x_2)
#        x_2 = F.relu(x_2)
#        x_2 = self.g2_fc3(x_2)
#        x_2 = F.relu(x_2)
        x_2 = self.g2_fc4(x_2)
        x_2 = F.dropout(x_2)
        x_2 = F.relu(x_2) # num_layers_value[b]*num_layers_value[b], 256
        x_2 = x_2.view(batch_size, MX_N, MX_N, 256)
#            print (x_)
        apps_vector_2 = []
        for b in range(batch_size):
            N = int(num_layers_value[b])
            x_2_tmp = x_2[b, :N, :N, :]
            x_2_tmp = x_2_tmp.contiguous().view(N*N, 256)
            x_2_tmp = x_2_tmp.sum(0).squeeze()
            apps_vector_2.append(x_2_tmp)
#        x_2 = x_2.sum(1).squeeze() # 256
        apps_vector_2 = torch.stack(apps_vector_2)
        
        """g3 """
        x_3 = self.g3_fc1(concat_vec_3)
        x_3 = F.relu(x_3)
#        x_3 = self.g3_fc2(x_3)
#        x_3 = F.relu(x_3)
#        x_3 = self.g3_fc3(x_3)
#        x_3 = F.relu(x_3)
        x_3 = self.g3_fc4(x_3)
        x_3 = F.dropout(x_3)
        x_3 = F.relu(x_3) # num_layers_value[b]*num_layers_value[b], 256
        x_3 = x_3.view(batch_size, MX_N, MX_N, MX_N, 256)
#            print (x_)
        apps_vector_3 = []
        for b in range(batch_size):
            N = int(num_layers_value[b])
            x_3_tmp = x_3[b, :N, :N, :N, :]
            x_3_tmp = x_3_tmp.contiguous().view(N*N*N, 256)
            x_3_tmp = x_3_tmp.sum(0).squeeze()
            apps_vector_3.append(x_3_tmp)
#        x_3 = x_3.sum(1).squeeze() # 256
        apps_vector_3 = torch.stack(apps_vector_3)

#        x_g = torch.cat([apps_vector_2, apps_vector_3], -1)
        x_g = torch.cat([apps_vector_2, apps_vector_3], -1)
        
#        print (x_g.size())
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f)

