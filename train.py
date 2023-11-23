import os
import shutil
import argparse
import numpy as np

from network.utils import *
from network.losses import wasserstein_loss, perceptual_loss
from network.model import generator, discriminator, DRGAN
from keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epoch_start", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epoch_num", type=int, default=200, help="number of training epoch")
parser.add_argument("--last_loss", type=float, default=float('inf'), help="last validation loss")
parser.add_argument("--train_num", type=int, default=200, help="number of training data")
parser.add_argument("--test_num", type=int, default=200, help="number of validation data")
parser.add_argument("--train_path", type=str, default="dataset/train/", help="path of the train dataset")
parser.add_argument("--test_path", type=str, default="dataset/test/", help="path of the test dataset")
parser.add_argument("--g_weight", type=str, default=None, help="path of the generator weight")
parser.add_argument("--d_weight", type=str, default=None, help="path of the discriminator weight")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=1e-4, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--ep", type=float, default=1e-08, help="adam: epsilon")
parser.add_argument("--img_size", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--save_interval", type=int, default=5, help="interval between saving image samples")
parser.add_argument("--lambda_gen", type=float, default=20, help="generator loss weight")
parser.add_argument("--lambda_dis", type=float, default=1, help="discriminator loss weight")
parser.add_argument("--critic_updates", type=int, default=3, help="Number of discriminator training")
parser.add_argument("--save_images", default='./experiments/imgs/', help="where to store images")
parser.add_argument("--save_models", default='./experiments/weights/', help="where to save models")
parser.add_argument("--gpu", type=str, default="2", help="gpu number")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.save_images, exist_ok=True)
os.makedirs(opt.save_models, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
def train_drgan():
   
    # construct models
    g = generator()
    d = discriminator()
    if opt.g_weight is not None:
        g.load_weights(opt.g_weight)
    if opt.d_weight is not None:
        d.load_weights(opt.d_weight)
    gan = DRGAN(g, d)

    # set up optimizer
    d_opt = Adam(learning_rate = opt.lr_g, beta_1 = opt.b1, beta_2 = opt.b2, epsilon = opt.ep)
    d_on_g_opt = Adam(learning_rate = opt.lr_d, beta_1 = opt.b1, beta_2 = opt.b2, epsilon = opt.ep)
    
    # compile models
    d.trainable = True
    d.compile(optimizer = d_opt, loss = wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [opt.lambda_gen, opt.lambda_dis]
    gan.compile(optimizer = d_on_g_opt, loss = loss, loss_weights = loss_weights)
    d.trainable = True
    
    # train
    best_val_loss = opt.last_loss
    for epoch in range(opt.epoch_start, opt.epoch_num):
        print('epoch: {}/{}'.format(epoch, opt.epoch_num))
        
        gan_loss = 0
        
        for _, (x_train_batch, y_train_batch) in enumerate(tqdm(load_batch(opt.batch_size, opt.train_num, opt.train_path), "Training", math.ceil(opt.train_num / opt.batch_size))):
            
            generated_images = g.predict(x = x_train_batch, batch_size = x_train_batch.shape[0], verbose=0)
            
            # labels for D
            output_true_batch, output_false_batch = np.ones((y_train_batch.shape[0], 1)), np.zeros((generated_images.shape[0], 1))

            for i in range(opt.critic_updates):    
                d.train_on_batch(y_train_batch, output_true_batch)
                d.train_on_batch(generated_images, output_false_batch)
            d.trainable = False
            gan_loss = gan.train_on_batch(x_train_batch, [y_train_batch, output_true_batch])
            d.trainable = True
            
        print("gan_loss: ", gan_loss)
        
        # quantitative evaluation
        val_loss = [0, 0, 0]
        val_count = 0
        sum_ssim_test = 0
        sum_psnr_test = 0
        for _, (x_test_batch, y_test_batch) in enumerate(tqdm(get_data(opt.batch_size, opt.test_num, opt.test_path), 'Validation', math.ceil(opt.test_num / opt.batch_size))):
            # loss calculation
            loss = gan.evaluate(x_test_batch, [y_test_batch, np.ones((y_test_batch.shape[0], 1))], x_test_batch.shape[0], verbose=0)
            val_loss = [val_loss[i] + loss[i] * x_test_batch.shape[0] for i in range(len(loss))]
            val_count += x_test_batch.shape[0]

            # metric calculation
            for i in range(x_test_batch.shape[0]):
                src_img_test = y_test_batch[i]
                rec_str_test = g(np.expand_dims(x_test_batch[i], 0), training=False).numpy()
                
                rec_img_test = np2img(rec_str_test)
                src_img_test = np2img(src_img_test)
                
                sum_ssim_test += ssim(src_img_test, rec_img_test, multichannel = True, channel_axis=2)
                sum_psnr_test += psnr(src_img_test, rec_img_test)

            # save image
            if ((epoch + 1) % opt.save_interval == 0):
                generated_images_test = g.predict(x = x_test_batch, batch_size = x_test_batch.shape[0], verbose=0)
                
                plot_images(x_test_batch, True, opt.save_images, "test_src", step = epoch)
                plot_images(y_test_batch, True, opt.save_images, "test_gt", step = epoch)
                plot_images(generated_images_test, True, opt.save_images, "test_pre", step = epoch)

        val_loss = [val_loss[i] / val_count for i in range(len(val_loss))]
        save_all_weights(g, d, opt.save_models, epoch, val_loss[0])
        if val_loss[0] < best_val_loss:
            best_val_loss = val_loss[0]
            for file in os.listdir('weights'):
                os.remove(f'weights/{file}')
            shutil.copytree('experiments/weights', 'weights', dirs_exist_ok=True)

            generated_images_test = g.predict(x = x_test_batch, batch_size = x_test_batch.shape[0], verbose=0)
                
            plot_images(x_test_batch, True, opt.save_images, "test_src", step = epoch)
            plot_images(y_test_batch, True, opt.save_images, "test_gt", step = epoch)
            plot_images(generated_images_test, True, opt.save_images, "test_pre", step = epoch)

        test_ssim = sum_ssim_test / val_count
        test_psnr = sum_psnr_test / val_count

        print("test ssim: ", test_ssim)
        print("test psnr: ", test_psnr) 
        
        f = open(opt.save_images + 'eva.txt', 'a')
        f.write('Epoch:' + str(epoch) + '\n')
        f.write('training loss:' + str(gan_loss) + '\n')
        f.write('validation loss:' + str(val_loss) + '\n')
        f.write('test ssim:' + str(test_ssim) + '\n')
        f.write('test psnr:' + str(test_psnr) + '\n')
        f.close()
    
if __name__ == '__main__':
    train_drgan()
