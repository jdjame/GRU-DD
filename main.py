import numpy as np 
from util import *
import sys, os, time, argparse, shutil, h5py, torch
from scipy.stats import pearsonr
import multiprocessing as mp
from og_gru import discModel
from og_gru import encodedGenerator
from gru_data import bkgdGen, gen_train_batch_bg, get1batch4test
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from kornia.losses import js_div_loss_2d,PSNRLoss,SSIMLoss
parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-depth',  type=int, default=1, help='input depth')
parser.add_argument('-maxep',  type=int, default=8000, help='max training epoches')
parser.add_argument('-warmup', type=int, default=100, help='warmup training epoches')
parser.add_argument('-mbsize', type=int, default=64, help='mini batch size')
parser.add_argument('-print',  type=str2bool, default=False, help='1:print to terminal; 0: redirect to file')
parser.add_argument('-logtrs', type=str2bool, default=False, help='log transform')
parser.add_argument('-sam',    type=str2bool, default=True, help='apply spatial attention')
parser.add_argument('-cam',    type=str2bool, default=True, help='apply channel attention')
parser.add_argument('-cvars',  type=str2list, default='T2:IWV:SLP', help='vars as condition')
parser.add_argument('-wmse',   type=float, default=5, help='weight of content loss for G loss')
parser.add_argument('-stage_chan',   type=int, default=4, help='stage channels for generator')
parser.add_argument('-lr_decay',   type=int, default=0, help='Should learning rate decay?')
parser.add_argument('-lr_disc',     type=float, default=3e-4, help='learning rate for discriminator')
parser.add_argument('-lr_gen',     type=float, default=3e-4, help='learning rate for generator')
parser.add_argument('-n_disc',   type=int, default=1, help='number of training steps for discriminator')
parser.add_argument('-n_gen',   type=int, default=1, help='number of training steps for generator')
parser.add_argument('-lam',   type=int, default=10, help='weight of GP')
parser.add_argument('-loss',   type=int, default=2, help='loss to use for content loss')
parser.add_argument('-add_loss',   type=int, default=0, help='loss to use for content loss')



args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch_devs)
print("log to %s, log Trans: %s, mb: %d, CAM: %s, SAM: %s, cvars:%s, wmse:%.1f" % (\
     'Terminal' if args.print else 'file', args.logtrs, args.mbsize,\
     args.cam, args.sam, ','.join(args.cvars), args.wmse))

itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

# redirect print to a file
if args.print == 0:
    sys.stdout = open(os.path.join(itr_out_dir, 'iter-prints.log'), 'w') 

def main(args):
    mb_size = args.mbsize
    in_depth = args.depth
    max_rain=31.63
    gene_model = encodedGenerator(in_ch=1, ncvar=3, stage_channels=[args.stage_chan])

    disc_model = discModel(in_ch=1, in_H=256, in_W=512)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            gene_model = torch.nn.DataParallel(gene_model)
            disc_model = torch.nn.DataParallel(disc_model)
        gene_model = gene_model.to(torch_devs)
        disc_model = disc_model.to(torch_devs)
        
    if args.loss==2:
        gene_criterion = torch.nn.MSELoss() 
    else:
        gene_criterion = torch.nn.L1Loss()
    if(args.add_loss):
        psnr_loss= PSNRLoss(max_rain)
        ssim_loss=SSIMLoss(5, max_rain)

    gene_optimizer = torch.optim.RMSprop(gene_model.parameters(), lr=args.lr_gen,alpha=0.9)
    disc_optimizer = torch.optim.RMSprop(disc_model.parameters(), lr=args.lr_disc,alpha=0.9)

    if(args.lr_decay):
        lrdecay_lambda = lambda epoch: cosine_decay(epoch, warmup=args.warmup, max_epoch=args.maxep)
        # initial lr times a given function
        gene_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gene_optimizer, lr_lambda=lrdecay_lambda)
        disc_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lrdecay_lambda)
        
    # build minibatch data generator with prefetch
    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(mb_size=mb_size, in_depth=in_depth, \
                                                             dev=torch_devs, trans=args.logtrs, cvars=args.cvars), \
                           max_prefetch=mb_size*4)

    ele_12km = h5py.File('../dataset/elevation_12km_resized.hdf5', "r")["elevation"]
    ele_12km = np.array([ele_12km] * mb_size)
    ele_12km = np.expand_dims(ele_12km, 1)
    ele_12km = torch.from_numpy(ele_12km).to(torch_devs)
   
    for epoch in range(args.maxep+1):
        time_it_st = time.time()
        X_mb, cvars, y_mb = mb_data_iter.next() # with prefetch
        
        # generator optimize
        for _ in range(args.n_gen):
            gene_optimizer.zero_grad()
            pred = gene_model.forward(X_mb, cvars, ele_12km)
            with torch.no_grad():
                advs_loss = 0 - disc_model.forward(pred,X_mb).mean()# adv loss
            cont_loss = gene_criterion(pred, y_mb) # content loss
            gene_loss = args.wmse * cont_loss + advs_loss
            if(args.add_loss):
                s_loss=ssim_loss(pred,y_mb) 
                p_loss=psnr_loss(pred,y_mb)
                gene_loss+=args.wmse *(s_loss+p_loss)
            gene_loss.backward()
            gene_optimizer.step() 
            if(args.lr_decay):
                gene_lr_scheduler.step()
    
        # discriminator optimize
        for _ in range(args.n_disc):
            disc_optimizer.zero_grad()
            disc_real_out=disc_model.forward(y_mb, X_mb)
            disc_fake_out=disc_model.forward(pred.detach(), X_mb)
            #gradient penalty
            epsilon=torch.rand(mb_size, 1,1,1).to(torch_devs)
            interp_image= y_mb * epsilon +  pred.detach()*(1-epsilon)
            interp_image.requires_grad=True
            interp_image=interp_image#.to(torch_devs)
            disc_interp_out= disc_model.forward(interp_image,X_mb)
            ones= torch.ones_like(disc_interp_out).to(torch_devs)
            grad_params = torch.autograd.grad(outputs=disc_interp_out,
                                              inputs=interp_image,
                                              create_graph=True,
                                              grad_outputs=ones)[0]
            grad_params = grad_params.reshape(grad_params.shape[0], -1)
            gradient_norm = grad_params.norm(2, dim=1)
            gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
            
            disc_loss = disc_fake_out.mean() - disc_real_out.mean() + args.lam * gradient_penalty
            disc_loss.backward()
            disc_optimizer.step()
            if(args.lr_decay):
                disc_lr_scheduler.step()
        if(args.add_loss):
            itr_prints = '[Info] @ %.1f Epoch: %05d, gloss: %.2f = (Cont:%.2f + SSIM:%2f + PSNR%2f + Adv:%.2f), dloss: %.2f, elapse: %.2fs/itr, gen_lr: %.5f, disc_lr: %.5f' % (\
                        time.time(), epoch, gene_loss.detach().cpu().numpy(), cont_loss.detach().cpu().numpy(),s_loss.detach().cpu().numpy(),p_loss.detach().cpu().numpy(), \
                        advs_loss.detach().cpu().numpy(), disc_loss.detach().cpu().numpy(), (time.time() - time_it_st), \
                        gene_optimizer.param_groups[0]['lr'],disc_optimizer.param_groups[0]['lr'])
        else:
            itr_prints = '[Info] @ %.1f Epoch: %05d, gloss: %.2f = (Cont:%.2f + Adv:%.2f), dloss: %.2f, elapse: %.2fs/itr, gen_lr: %.5f, disc_lr: %.5f' % (\
                        time.time(), epoch, gene_loss.detach().cpu().numpy(), cont_loss.detach().cpu().numpy(), \
                        advs_loss.detach().cpu().numpy(), disc_loss.detach().cpu().numpy(), (time.time() - time_it_st), \
                        gene_optimizer.param_groups[0]['lr'],disc_optimizer.param_groups[0]['lr'])
        print(itr_prints)

        if epoch % (10) == 0:
            if epoch == 0: 
                X222, cv222, y222 = get1batch4test(in_depth=in_depth, idx=range(args.mbsize), dev=torch_devs, \
                                            trans=args.logtrs, cvars=args.cvars)
                save2img_rgb(torch.squeeze(X222)[0,in_depth-1,:,:].cpu(), '%s/low-res.png' % (itr_out_dir))
                true_img = y222.cpu().numpy()
                if args.logtrs: true_img = np.exp(true_img) - 1 # transform back
                save2img_rgb(true_img[0,0,:,:], '%s/high-res.png' % (itr_out_dir))

            with torch.no_grad():
                pred_img = gene_model.forward(X222, cv222, ele_12km).cpu().numpy()
                if args.logtrs: pred_img = np.exp(pred_img) - 1 # transform back
                mse = np.mean((true_img - pred_img)**2)
                cc_avg = np.mean([pearsonr(pred_img[i].flatten(), true_img[i].flatten())[0] for i in range(pred_img.shape[0])])
                # SSIM
                ssim_score = ssim( torch.tensor(pred_img), torch.tensor(true_img) , data_range=max_rain, size_average=False ).mean().numpy()
                #MS SSIM
                ms_ssim_score= ms_ssim( torch.tensor(pred_img), torch.tensor(true_img) , data_range=max_rain, size_average=False ).mean().numpy()
                #NSE
                nse_score = 1-nse_loss(gen=torch.tensor(pred_img), obs=torch.tensor(true_img))#np.mean([nse(pred, true) for pred,true in zip(pred_img_tmp,true_img_tmp)])
                #PSNR
                max_i=np.max(true_img, axis=(1,2,3))**2
                mse_i=np.mean((true_img-pred_img)**2, axis=(1,2,3))
                psnr = np.mean(10*np.log10(max_i/mse_i))
                js=js_div_loss_2d(torch.tensor(pred_img), torch.tensor(true_img)).item()
            print('[Validation] @ Epoch: %05d MSE: %.4f, CC: %.3f , SSIM: %.3f , MS-SSIM: %.3f , NSE: %.3f , PSNR: %.3f, JS: %.3f of %d samples' % (epoch, mse, cc_avg, ssim_score, ms_ssim_score, nse_score, psnr, js,pred_img.shape[0]))

            if epoch%1000==0:
                
                save2img_rgb(pred_img[0,0,:,:], '%s/it%05d.png' % (itr_out_dir, epoch))

                if torch.cuda.device_count() > 1:
                    torch.save(gene_model.module.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
                else:
                    torch.save(gene_model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))

        sys.stdout.flush()

def nse_loss(gen, obs):
    num=torch.sum((gen-obs)**2, dim=0)
    mean_obs= torch.mean(obs,dim=0)
    den=torch.sum(torch.sub(obs,mean_obs[None,:,:,:])**2, dim=0)+ 1e-5
    glob_nse=num/den
    glob_nse= 1-1/(1+glob_nse)
    mean_nse= torch.mean(glob_nse)
    
    return mean_nse


if __name__ == '__main__':
    main(args)
