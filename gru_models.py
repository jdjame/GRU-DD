import torch, torchvision
from torch import nn
import numpy as np

# Channel attention module 
class CAM(torch.nn.Module):
    def __init__(self, in_ch, ratio=2):
        super(CAM, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
           
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x*self.sigmoid(out)

# Spatial attention module 
class SAM(torch.nn.Module):
    def __init__(self, in_ch,relu_a=0.01):
        super().__init__()
        self.cnn_ops = [
            torch.nn.Conv2d(in_channels=2, out_channels=1, \
                            kernel_size=7, padding='same'),
            torch.nn.Sigmoid(), ] # use Sigmoid to norm to [0, 1]
        
        self.attention_layer = torch.nn.Sequential(*self.cnn_ops)
        
    def forward(self, x, ret_att=False):
        _max_out, _ = torch.max(x, 1, keepdim=True)
        _avg_out    = torch.mean(x, 1, keepdim=True)
        _out = torch.cat((_max_out, _avg_out), dim=1)
        _attention = _out
        for layer in self.attention_layer:
            _attention = layer(_attention)
           
        if ret_att:
            return _attention, _attention * x
        else:
            return _attention * x
        
class TimeDistributed(torch.nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1,x.size(-3), x.size(-2),x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-3),y.size(-2),y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class Up_block(torch.nn.Module):
    def __init__(self, in_ch, scale_factor,channel_factor):
        super(Up_block,self).__init__()
        
        self.ident = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(in_ch, in_ch //channel_factor, kernel_size=(1,1),padding='same')
            )
        )
        self.fmap = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=scale_factor),
            torch.nn.Conv2d(in_ch, in_ch // channel_factor, kernel_size=(3,3),padding='same'),
            torch.nn.BatchNorm2d(in_ch // channel_factor),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_ch // channel_factor, in_ch //channel_factor, kernel_size=(3,3),padding='same')
        )
    
    def forward(self,x):
        ident=self.ident(x)
        fmap=self.fmap(x)
        return ident+fmap
    
# Downsampling Block
class D_block(torch.nn.Module):
    def __init__(self, in_ch,in_H=0, in_W=0,reluFirst=True, factor=2, downsample=True):
        
        super(D_block,self).__init__()
        
        # bool flag for first relu activation
        self.reluFirst= reluFirst
        
        #identity block of resnet
        identity_layers=[torch.nn.Conv2d(in_ch,in_ch*factor,kernel_size=(1,1),padding='same')]
        if (downsample):
            identity_layers.append(torch.nn.AvgPool2d(kernel_size=(2,2)))
        self.ident = torch.nn.Sequential( *identity_layers )
        
        #functional mapping for resnet
        func_map_layers=[torch.nn.Conv2d(in_ch, in_ch *factor, kernel_size=(3,3),padding='same'),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_ch*factor , in_ch *factor, kernel_size=(3,3),padding='same')]
        if (downsample):
            func_map_layers.append(torch.nn.AvgPool2d(kernel_size=(2, 2)))
        self.fmap = torch.nn.Sequential(*func_map_layers)
                            
    
    def forward(self,x):
        ident= self.ident(x)
        func_map=x
        if (self.reluFirst):
            func_map= torch.nn.ReLU()(func_map)
        func_map=self.fmap(func_map)
        
        return ident+func_map



# residual blocks 
class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch,stride=1  ):
        super().__init__()
        
        short_con= []
        
        
        if stride>1:
            short_con.append(torch.nn.AvgPool2d(stride))
        if (in_ch != out_ch):
            short_con.append(torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                             kernel_size=1, stride=1))
        long_con= [torch.nn.LeakyReLU(0.2) , 
                   torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                   kernel_size=3, stride=stride, padding='same'),
                   torch.nn.LeakyReLU(0.2),
                   torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                                   kernel_size=3, stride=1, padding='same'),
                   torch.nn.LeakyReLU(0.2)]
        
        self.short_con= torch.nn.Sequential(*short_con)
        self.long_con= torch.nn.Sequential(*long_con)
        
    def forward(self, x):
        return self.short_con(x)+ self.long_con(x)

class ConvGRUCell(torch.nn.Module):
    
    def __init__(self,input_size,hidden_size,kernel_size,cuda_flag=True):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = torch.nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,3,padding='same')
        self.Conv_ct     = torch.nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,3,padding='same') 
        dtype            = torch.FloatTensor
    
    def forward(self,input,hidden):
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           hidden    = torch.autograd.Variable(torch.zeros(size_h)).to(input.device)
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = torch.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h

class encodedGenerator(torch.nn.Module):

    def __init__(self, in_ch=1, ncvar=3, relu_a=0.01, use_ele=True, cam=True, sam=True, stage_channels=[256]):
        super().__init__()
        self.in_ch = in_ch
        self.use_sam = sam
        self.use_cam = cam
        self.rain_res= torch.nn.Sequential(*[TimeDistributed(torch.nn.Conv2d(in_channels=in_ch, out_channels= stage_channels[0],
                                                                             kernel_size=3, padding='same')),
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0])), 
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0])),
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0]))]
                                           )
        self.hidden_0= torch.nn.Sequential(*[TimeDistributed(torch.nn.Conv2d(in_channels=in_ch, out_channels= stage_channels[0]*(ncvar+1),
                                                                             kernel_size=3, padding='same')),
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0]*(ncvar+1), out_ch=stage_channels[0]*(ncvar+1))), 
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0]*(ncvar+1), out_ch=stage_channels[0]*(ncvar+1))),
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0]*(ncvar+1), out_ch=stage_channels[0]*(ncvar+1)))]
                                           )
        self.cvar_res= [torch.nn.Sequential(*[TimeDistributed(torch.nn.Conv2d(in_channels=in_ch, out_channels= stage_channels[0],
                                                                             kernel_size=3, padding='same')),
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0])), 
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0])),
                                             TimeDistributed(ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0]))]
                                           ) for _ in range(ncvar)]
        self.cvar_res = torch.nn.ModuleList(self.cvar_res)
        
        gru_factors=[1,2,4]
        self.gru_cells= torch.nn.ModuleList([ConvGRUCell(input_size=stage_channels[0]*(ncvar+1)//fact,
                                     hidden_size=stage_channels[0]*(ncvar+1)//fact,kernel_size=3) for fact in gru_factors])
        
        self.up_blocks_h= torch.nn.ModuleList([Up_block(in_ch=stage_channels[0]*(ncvar+1)//fact, scale_factor=2, channel_factor=2) for fact in gru_factors])
        self.up_blocks_x= torch.nn.ModuleList([Up_block(in_ch=stage_channels[0]*(ncvar+1)//fact, scale_factor=2, channel_factor=2) for fact in gru_factors])
        
        self.num_grus=len(self.gru_cells)
        
        cur_channels = stage_channels[0]*(ncvar+1) //(2**self.num_grus) # B x cur_channels x 2^num_grus*H x 2^num_grus*W
        
        added=stage_channels[0]//gru_factors[-1] if use_ele else 0

        #attention modules
        if self.use_cam:
            self.cam1 = TimeDistributed(CAM(in_ch =stage_channels[0]*(ncvar+1)))
            self.cam2 = CAM(in_ch = cur_channels+added)

        if self.use_sam:
            self.sam1= TimeDistributed(SAM(in_ch=stage_channels[0]*(ncvar+1)))
            self.sam2= SAM(in_ch=cur_channels+added)
            
        #elevation module 
        if use_ele:
            self.ele_layers = torch.nn.Sequential(*[torch.nn.Conv2d(in_channels=in_ch, out_channels= stage_channels[0],
                                                                             kernel_size=3, padding='same'), 
                                                    ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0]),
                                                    ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0]),
                                                    ResBlock(in_ch=stage_channels[0], out_ch=stage_channels[0]//gru_factors[-1]),
                                                    Up_block(in_ch=stage_channels[0]//gru_factors[-1], scale_factor=2, channel_factor=1)])
        
        #final operations
        self.out_ops=torch.nn.Sequential(*[torch.nn.Conv2d(in_channels= cur_channels+added, out_channels=1, kernel_size=1 ), 
                                           D_block(in_ch=1, factor=1,downsample=True),
                                           D_block(in_ch=1, factor=1,downsample=False),
                                           D_block(in_ch=1, factor=1,downsample=False),
                                           torch.nn.ReLU()])


    def forward(self, x, cvars, elevation=None):
        # x is assumed to be B x T x C x H x W
        #pre process cvars 
        cvar_pp = []
        for _cf, cvar in zip(self.cvar_res, cvars):
            _tmp = cvar
            for _f in _cf:
                _tmp = _f(_tmp)
            cvar_pp.append(_tmp)
            
        #pre process rain
        rain =self.rain_res(x) 
        
        #concatenate the processed variables accross channel axis 
        rain = torch.cat([rain,] + cvar_pp, 2)
        
        #####optional add channel attention and spatial attention here (later)###
        if self.use_sam:
            rain=self.sam1(rain)
        if(self.use_cam):
            rain=self.cam1(rain)
        hidden_cond_frame=x.shape[1]-1#changed from 0
        #greate hidden_0 for gru
        hidden_0= torch.squeeze(self.hidden_0(x[:,hidden_cond_frame,:,:]))
        
        #Convolutional GRU 
        hidden_next=hidden_0
        stored_hidden=[hidden_next for _ in range(self.num_grus+1)]
        for t in range(x.shape[1]): #iterate through time steps 
            
            stored_x=[rain[:,t,:,:] for _ in range(self.num_grus+1)]
            for i,mods in enumerate(zip(self.gru_cells, self.up_blocks_x,self.up_blocks_h )):
                
                _gru,_up_samp_x, _up_samp_h = mods
                #retrieve current hidden and x 
                tmp_x= stored_x[i]
                tmp_hidden= stored_hidden[i]
                #gru calculation
                hidden_next= _gru(tmp_x, tmp_hidden)
                #overwrite hidden for next time step
                stored_hidden[i]=hidden_next
                #overwrite stored hidden for next gru
                stored_hidden[i+1]=_up_samp_h(hidden_next)
                #overwrite stored x for next gru
                stored_x[i+1]= _up_samp_x(tmp_x)
        
        opt_out= stored_hidden[-1]
        
        #incoorporate evelvation
        elevation=self.ele_layers(elevation)

        opt_out = torch.cat([opt_out, elevation], 1)
        if self.use_sam:
            opt_out=self.sam2(opt_out)
        if(self.use_cam):
            opt_out=self.cam2(opt_out)
            
        return self.out_ops(opt_out)
    

class discModel(torch.nn.Module):
    def __init__(self,in_ch=1, in_H=256, in_W=512, use_cam=True, use_sam=True):
        super().__init__()
        self.in_ch=in_ch
        self.in_H=in_H
        self.in_W=in_W
        
        self.use_cam=use_cam
        self.use_sam=use_sam
            
        high_res_ops=[2,2]
        high_res_pp= [D_block(in_ch=in_ch, in_H= in_H, in_W=in_W, reluFirst=False, factor=high_res_ops[0], downsample=True)]
        hr_cur_ch=in_ch*high_res_ops[0]
        high_res_pp.append(D_block(in_ch=in_ch*2, in_H= in_H//2, in_W=in_W//2, reluFirst=False, factor=high_res_ops[1], downsample=True))
        hr_cur_ch=hr_cur_ch*high_res_ops[1]
        self.high_res_pp=torch.nn.Sequential(*high_res_pp)
        
        self.low_res_pp= torch.nn.Sequential(*[TimeDistributed(ResBlock(in_ch=in_ch, out_ch=high_res_ops[0])),
                                              TimeDistributed(ResBlock(in_ch=high_res_ops[0], out_ch=high_res_ops[0]*high_res_ops[1]))])
        lr_cur_ch= high_res_ops[0]*high_res_ops[1]
        assert lr_cur_ch==hr_cur_ch
        self.merged_res=torch.nn.Sequential(*[TimeDistributed(ResBlock(in_ch=hr_cur_ch, out_ch=hr_cur_ch)),
                                              TimeDistributed(ResBlock(in_ch=hr_cur_ch, out_ch=hr_cur_ch)),
                                              TimeDistributed(ResBlock(in_ch=hr_cur_ch, out_ch=hr_cur_ch))])
        #gru + space to depth modules
        growth=[1,2,4]
        self.merged_grus= torch.nn.ModuleList([ConvGRUCell(input_size=hr_cur_ch*fact,
                                       hidden_size=hr_cur_ch*fact,kernel_size=3) for fact in growth])
        
        #space to depth module 
        self.downscalers_h= torch.nn.ModuleList([D_block(in_ch=hr_cur_ch*fact,in_H=256//fact, in_W=512//fact) for fact in growth])
        self.downscalers_x= torch.nn.ModuleList([D_block(in_ch=hr_cur_ch*fact,in_H=256//fact, in_W=512//fact) for fact in growth])
        
        self.num_grus=len(growth)
        
        cur_ch=hr_cur_ch*growth[-1]*2
        
        if use_sam:
            self.sam1=SAM(in_ch=cur_ch)
            self.sam2=SAM(in_ch =lr_cur_ch)
        if use_cam:
            self.cam1=CAM(in_ch =cur_ch)
            self.cam2=CAM(in_ch =lr_cur_ch)
            
        # convolutions for the high res sequence 
        self.complete_hr= torch.nn.Sequential(*[D_block(in_ch=hr_cur_ch*fact,in_H=256//fact, in_W=512//fact) for fact in growth])
        
        final_h=in_H
        final_w= in_W
        for i in growth:
            final_h=final_h//i
            final_w=final_w//i
        final_h=final_h//4
        final_w=final_w//4
        self.predictive_block= torch.nn.Sequential(*[torch.nn.Flatten(),
                                                     torch.nn.Linear(cur_ch*2 *final_h*final_w , 4), 
                                                     torch.nn.LeakyReLU(0.2),
                                                     torch.nn.Linear(4,1)])
    def forward(self, high_res_x, low_res_x):
        
        #preprocess high res 
        high_x= self.high_res_pp(high_res_x)
        #preprocess low rres
        low_x= self.low_res_pp(low_res_x)
        #append lr map to hr map
        high_x_temp=high_x[:,None, :,:]
        merged_x= torch.cat([low_x,high_x_temp], 1)
        #process joined mapping
        merged_x=self.merged_res(merged_x)
        
        #use gru on mapping
        stored_hidden=[None for _ in range(self.num_grus+1)]
        for t in range(low_res_x.shape[1]+1):
            stored_x=[merged_x[:,t,:,:] for _ in range(self.num_grus+1)]
            for i,mods in enumerate(zip(self.merged_grus,self.downscalers_x,self.downscalers_h)):
                _gru, _dsx, _dsh = mods
                
                #retrieve current hidden and x 
                tmp_x= stored_x[i]
                tmp_hidden= stored_hidden[i]
                #gru calculation
                hidden_next= _gru(tmp_x, tmp_hidden)
                #overwrite hidden for next time step
                stored_hidden[i]=hidden_next
                #overwrite stored hidden for next gru
                stored_hidden[i+1]=_dsh(hidden_next)
                #overwrite stored x for next gru
                stored_x[i+1]= _dsx(tmp_x)
        opt_out= stored_hidden[-1]
        if self.use_sam:
            opt_out=self.sam1(opt_out)
            high_x=self.sam2(high_x)
        if self.use_cam:
            opt_out=self.cam1(opt_out)
            high_x=self.cam2(high_x)
            
        high_x=self.complete_hr(high_x)
        
        final_out= torch.cat([high_x, opt_out], 1)
        return self.predictive_block(final_out)