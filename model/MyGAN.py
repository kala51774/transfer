from torch import nn
import torch
import time as t
from tqdm import tqdm
from torch import optim
from torchvision import transforms
# from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from torch.utils import data
from torch.nn.functional import interpolate

from Discriminator.Discriminator_S import Discriminator_S
from Encoder.StyleEncoder import StyleEncoder
from model.Discriminator_T import Discriminator_T
from model.Generator import Generator
from model.VGG19 import VGG19
from utils.ImagePools import ImagePools
from utils.content_struct import content_struct
from  utils.funs import *
from torch.utils.tensorboard import SummaryWriter
class MyGAN(object):
    def __init__(self, args):
        super().__init__()
        # 定义配置
        self.logdir = './runs'.join(self.dataset) if args.logdir==None else args.logdir
        self.iter=args.iter

        self.writer = SummaryWriter(log_dir=args.logdir)
        self.device = args.device
        self.result_dir = args.result_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.test_dir = args.test_dir
        self.isTrain = args.isTrain
        self.isTest = args.isTest
        self.train_init = args.train_init
        self.epoch = args.epoch
        self.cur_epoch=0
        self.pre_epoch = args.pre_epoch
        self.cpu_count = args.cpu_count
        # 定义模型参数
        self.input_c = args.input_c
        self.hw = args.hw
        self.b1 = args.b1
        self.b2 = args.b2
        self.y1 = args.y1
        self.y2 = args.y2
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.latent_dim = args.latent_dim
        self.pk = args.patch_size
        self.s = args.s
        self.decay_lr = args.decay_lr
        self.init_lr = args.init_lr
        self.batch_size = args.batch_size
        self.save_pred = args.save_pred
        # 模型权重参数w
        self.weight_content = args.weight_content
        self.weight_surface = args.weight_surface
        self.weight_testure = args.weight_testure
        self.weight_struct = args.weight_struct

        self.weight_style = args.weight_style
        self.weight_decay = args.weight_decay
        self.tv_weight = args.weight_tv
        # 定义模型
        self.G = Generator().to(self.device)
        self.D = Discriminator_S().to(self.device)
        self.D_patch = Discriminator_T().to(self.device)
        self.style_net = StyleEncoder().to(self.device)
        # self.sct = utm().to(self.device)
        self.sct = args.neck().to(self.device)

        self.vgg19 = VGG19().to(self.device)
        self.vgg19.eval()
        self.p = content_struct().to(self.device)
        # 模型初始化
        self.G.apply(init_weights)
        self.D.apply(init_weights)
        self.D_patch.apply(init_weights)
        self.sct.apply(init_weights)
        self.style_net.apply(init_weights)
        self.vgg19.load_state_dict(torch.load('vgg19.pth'))
        # 定义优化器
        if self.train_init:
            self.optim_G = optim.Adam(self.G.parameters(), lr=self.init_lr, betas=(self.b1, self.b2))
        else:
            self.optim_G = optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.b1, self.b2))
        self.op_style_net = optim.Adam(self.style_net.parameters(), lr=self.d_lr, betas=(self.b1, self.b2))
        self.optim_sct = optim.Adam(self.sct.parameters(), lr=self.d_lr, betas=(self.b1, self.b2))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.b1, self.b2))
        self.optim_D_Patch = optim.Adam(self.D_patch.parameters(), lr=self.g_lr, betas=(self.b1, self.b2))
        # 定义损失
        self.l1_loss = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self.gan_loss = nn.MSELoss()
        self.tv_loss = VariationLoss(1)
        self.lsty = nn.CrossEntropyLoss()
        self._rgb_to_yuv_kernel = torch.tensor([
            [0.299, -0.14714119, 0.61497538],
            [0.587, -0.28886916, -0.51496512],
            [0.114, 0.43601035, -0.10001026]
        ]).float().to(self.device)
        # 辅助器).to(self.device)
        self.gf = GuidedFilter()
        self.query = Queue()
        self.edge_exactor = CannyFilter().to(self.device)
        # 打印配置
        print("##### Information #####")
        print("# device : ", self.device)
        print(f"线程:{int(self.cpu_count)}")
        print("# dataset : ", self.dataset)
        print(f'# G_num :{print_network(self.G)}')
        print(f'# D_num :{print_network(self.D)}')
        print(f'# D_Patch_num :{print_network(self.D_patch)}')
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# pre epoch : ", self.pre_epoch)
        print("# init_train : ", self.train_init)
        print("# training image size [H, W] : ", self.hw)
        print("# content,style,suface,testure,color_weight,tv_weight: ", self.weight_content, self.weight_style,
              self.weight_surface,
              self.weight_testure, self.weight_struct, self.tv_weight)
        print("# init_lr,g_lr,d_lr: ", self.init_lr, self.g_lr, self.d_lr)

    # 生成假图.to(self.device)

    # 读取数据集
    def load_data(self, epoch_test=False, high=False):
        self.mode = epoch_test
        self.high = high
        # 增强操作
        train_trans = [transforms.Resize(286),
                       transforms.CenterCrop(256),
                       transforms.RandomHorizontalFlip(0.5),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        test_trans = [transforms.Resize([512, 512]),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if self.mode and self.isTrain:
            data_loader = data.DataLoader(ImagePools(root=self.data_dir, trans=test_trans, mode=self.mode),
                                          batch_size=4,
                                          pin_memory=True,
                                          drop_last=True
                                          , num_workers=0)
        if self.isTrain and not self.mode:
            data_loader = data.DataLoader(ImagePools(root=self.data_dir, trans=train_trans),
                                          batch_size=self.batch_size, pin_memory=False,
                                          drop_last=True
                                          , num_workers=0)

        if self.isTest and self.mode:
            if self.high:
                data_loader = data.DataLoader(
                    ImagePools(root=self.data_dir, trans=test_trans, mode=self.mode, high=self.high), batch_size=1,
                    num_workers=2)
            else:
                data_loader = data.DataLoader(ImagePools(root=self.data_dir, trans=test_trans, mode=self.mode),
                                              batch_size=1, num_workers=2)
        if self.isTest and not self.mode:
            data_loader = data.DataLoader(ImagePools(root=self.data_dir, trans=test_trans), batch_size=1, num_workers=2)
        return data_loader

    # 加载灰度patch
    def load_patch(self, real, fake):
        # 进行灰度颜色改变
        real_gry = color_shift(real)
        fake_gry = color_shift(fake)
        return real_gry, fake_gry

    # conten loss
    def content_loss(self, fake, real):
        _, c, w, h = fake.shape
        out_con = self.l1_loss(fake, real)
        return out_con

    # dis loss
    def discriminator_loss(self, real, fake):
        real_loss = torch.mean(torch.square(real - 1.0))
        fake_loss = torch.mean(torch.square(fake))
        loss = real_loss + fake_loss
        return loss

    def generator_loss(self, fake):
        fake_loss = torch.mean(torch.square(fake - 1.0))
        return fake_loss

    # dis loss
    def discriminator_gram_loss(self, real, fake):
        real = gram(real)
        fake = gram(fake)
        real_loss = torch.mean(torch.square(real - 1.0))
        fake_loss = torch.mean(torch.square(fake))
        loss = real_loss + fake_loss
        return loss

    def generator_gram_loss(self, fake):
        fake = gram(fake)
        fake_loss = torch.mean(torch.square(fake - 1.0))
        return fake_loss

    # 训练
    def train(self):
        data_loader = self.load_data()
        count = len(data_loader)

        start_t = t.time()
        self.G.train()
        if self.train_init:
            print("=============================pre train phase==============================")
            for epoch in tqdm(range(self.pre_epoch)):
                for i, (x, y) in enumerate(data_loader):
                    self.iter+=1
                    x, y = x.to(self.device), y.to(self.device)
                    # 预训练阶段
                    requires_grad(self.D, False)
                    requires_grad(self.D_patch, False)
                    requires_grad(self.style_net, False)
                    self.optim_G.zero_grad()
                    self.optim_sct.zero_grad()
                    con_code = self.G.encoder(x)
                    con_code = self.sct(con_code, init=True)
                    fake1_img = self.G.decoders(con_code)
                    with torch.no_grad():
                        real_con = self.vgg19(x)
                        fake_con1 = self.vgg19(fake1_img)
                    con_loss_1 = self.content_loss(fake_con1, real_con.detach())
                    # 128
                    real_con2 = interpolate(x, scale_factor=0.5, mode='bilinear')
                    real_con2 = interpolate(real_con2, scale_factor=2, mode='bilinear')
                    fake_con2 = interpolate(fake1_img, scale_factor=0.5, mode='bilinear')
                    fake_con2 = interpolate(fake_con2, scale_factor=2, mode='bilinear')
                    con_loss_2 = self.content_loss(fake_con2, real_con2.detach())
                    # 64
                    real_con3 = interpolate(x, scale_factor=0.25, mode='bilinear')
                    real_con3 = interpolate(real_con3, scale_factor=4, mode='bilinear')
                    fake_con3 = interpolate(fake1_img, scale_factor=0.25, mode='bilinear')
                    fake_con3 = interpolate(fake_con3, scale_factor=4, mode='bilinear')
                    con_loss_3 = self.content_loss(fake_con3, real_con3.detach())
                    con_loss = (con_loss_1 + con_loss_2 + con_loss_3) / 3 * self.weight_content

                    self.writer.add_scalar('con_loss1', con_loss_1, self.iter)
                    self.writer.add_scalar('con_loss2', con_loss_2, self.iter)
                    self.writer.add_scalar('con_loss3', con_loss_3, self.iter)
                    self.writer.add_scalar('con_loss', con_loss, self.iter)
                    self.iter += 1

                    con_loss.backward()
                    self.optim_G.step()
                    self.optim_sct.step()
                    end_epoch_t = t.time()



                    print(
                        f"epoch:[{epoch + 1}/{self.pre_epoch}],iter:[{i + 1}/{count}],loss_G:{con_loss},G_lr:{self.optim_G.param_groups[0]['lr']},"
                        f"time:{time_change(end_epoch_t - start_t)}")
                if epoch % self.save_pred == 0:
                    self.save_img1(epoch)
        else:
            print('==========================start train=====================================')
            for epoch in tqdm(range(self.epoch)):

                # 学习率衰退
                if epoch > 99:
                    self.optim_G.param_groups[0]['lr'] -= 0.0002 / 50
                    self.optim_D.param_groups[0]['lr'] -= 0.0002 / 50
                    self.optim_D_Patch.param_groups[0]['lr'] -= 0.0002 / 50
                    self.optim_sct.param_groups[0]['lr'] -= 0.0002 / 50
                    self.op_style_net.param_groups[0]['lr'] -= 0.0002 / 50
                for i, (x, y) in enumerate(data_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    self.D.train(), self.D_patch.train()
                    # zero grident
                    self.optim_D.zero_grad()
                    self.optim_D_Patch.zero_grad()
                    self.op_style_net.zero_grad()
                    self.optim_sct.zero_grad()
                    # D
                    style_code = self.style_net(y)
                    content_code = self.G.encoder(x)
                    share_code = self.sct(content_code, style_code)
                    fake_img = self.G.decoders(share_code)
                    # 减少模型震荡
                    # surface
                    gf_real_img_h = interpolate(y, scale_factor=2, mode='bilinear')
                    gf_fake_img_h = interpolate(fake_img, scale_factor=2, mode='bilinear')
                    gf_real_img_h = self.gf.guided_filter(gf_real_img_h, gf_real_img_h, r=5, eps=2e-1)
                    gf_fake_img_h = self.gf.guided_filter(gf_fake_img_h, gf_fake_img_h, r=5, eps=2e-1)

                    d_real_logit = self.D(gf_real_img_h)
                    d_fake_logit = self.D(gf_fake_img_h.detach())
                    d_surface_loss = self.discriminator_loss(d_real_logit, d_fake_logit)
                    # testure
                    anime_gry_patch, fake_gry_patch = self.load_patch(gf_real_img_h, gf_fake_img_h)
                    #                     anime_gry_patch = interpolate(anime_gry_patch,scale_factor=2,mode='bilinear')
                    #                     fake_gry_patch = interpolate(fake_gry_patch,scale_factor=2,mode='bilinear')
                    real_patch_logit = self.D_patch(anime_gry_patch)
                    fake_patch_logit = self.D_patch(fake_gry_patch.detach())
                    d_testure_loss = self.discriminator_gram_loss(real_patch_logit, fake_patch_logit)
                    d_loss = (d_surface_loss + d_testure_loss) / 2
                    d_loss.backward()
                    self.optim_D.step()
                    self.optim_D_Patch.step()
                    # G
                    self.style_net.train()
                    self.sct.train()
                    self.optim_G.zero_grad()
                    self.op_style_net.zero_grad()
                    self.optim_sct.zero_grad()
                    # style  loss
                    style_code = self.style_net(y)
                    content_code = self.G.encoder(x)
                    share_code = self.sct(content_code, style_code)
                    fake1_img = self.G.decoders(share_code)
                    # surface
                    gf_fake_img_h1 = interpolate(fake1_img, scale_factor=2, mode='bilinear')
                    #                     y =interpolate(y,scale_factor=2,mode='bilinear')
                    gf_fake_img_h1 = self.gf.guided_filter(gf_fake_img_h1, gf_fake_img_h1, r=5, eps=2e-1)

                    g_fake_logit1 = self.D(gf_fake_img_h1)
                    g_surface_loss = self.generator_loss(g_fake_logit1) * self.weight_surface
                    # testure
                    _, fake_gry_patch1 = self.load_patch(gf_fake_img_h1, gf_fake_img_h1)
                    #                     fake_gry_patch1 = interpolate(fake_gry_patch1,scale_factor=2,mode='bilinear')
                    fake_patch_logit = self.D_patch(fake_gry_patch1)

                    g_testure_loss = self.generator_gram_loss(fake_patch_logit) * self.weight_testure
                    # multi-level content loss
                    with torch.no_grad():
                        real_con = self.vgg19(x)
                        fake_con1 = self.vgg19(fake1_img)
                    con_loss_1 = self.content_loss(fake_con1, real_con.detach())
                    # 128
                    real_con2 = interpolate(x, scale_factor=0.5, mode='bilinear')
                    real_con2 = interpolate(real_con2, scale_factor=2, mode='bilinear')
                    fake_con2 = interpolate(fake1_img, scale_factor=0.5, mode='bilinear')
                    fake_con2 = interpolate(fake_con2, scale_factor=2, mode='bilinear')
                    con_loss_2 = self.content_loss(fake_con2, real_con2.detach())
                    # 64
                    real_con3 = interpolate(x, scale_factor=0.25, mode='bilinear')
                    real_con3 = interpolate(real_con3, scale_factor=4, mode='bilinear')
                    fake_con3 = interpolate(fake1_img, scale_factor=0.25, mode='bilinear')
                    fake_con3 = interpolate(fake_con3, scale_factor=4, mode='bilinear')
                    con_loss_3 = self.content_loss(fake_con3, real_con3.detach())
                    con_loss = (con_loss_1 + con_loss_2 + con_loss_3) / 3 * self.weight_content
                    # tv loss
                    tv_loss = self.tv_loss(fake1_img)
                    # color re LOSS
                    col_real_img = rgb_to_yuv(x, self._rgb_to_yuv_kernel)
                    col_fake_img = rgb_to_yuv(fake1_img, self._rgb_to_yuv_kernel)
                    col_loss = 10 * (
                            self.l1_loss(col_real_img[:, 0, :, :], col_fake_img[:, 0, :, :]) + self.huber(
                        col_real_img[:, 1, :, :], col_fake_img[:, 1, :, :]) + \
                            self.huber(col_real_img[:, 2, :, :], col_fake_img[:, 2, :, :]))

                    g_loss = (g_surface_loss + g_testure_loss + con_loss + tv_loss + col_loss) / 5
                    #                     else:
                    g_loss.backward()
                    self.optim_G.step()
                    self.op_style_net.step()
                    self.optim_sct.step()
                    end_epoch_t = t.time()
                    self.writer.add_scalar('tv_loss1', tv_loss, self.iter)
                    self.writer.add_scalar('g_surface_loss', g_surface_loss, self.iter)
                    self.writer.add_scalar('g_testure_loss', g_testure_loss, self.iter)
                    self.writer.add_scalar('col_loss', col_loss, self.iter)
                    self.writer.add_scalar('con_loss1', con_loss_1, self.iter)
                    self.writer.add_scalar('con_loss2', con_loss_2, self.iter)
                    self.writer.add_scalar('con_loss3', con_loss_3, self.iter)
                    self.writer.add_scalar('con_loss', con_loss, self.iter)
                    self.iter += 1


                    print(
                        f"epoch:[{epoch + 1}/{self.epoch}],iter:[{i + 1}/{count}],loss_G:{g_loss},loss_d:{d_loss},G_lr:{self.optim_G.param_groups[0]['lr']},D_lr:{self.optim_D.param_groups[0]['lr']},time:{time_change(end_epoch_t - start_t)}")
                if (epoch + 1) % self.save_pred == 0:
                    with torch.no_grad():
                        self.save_model()

    #                          self.save_img(epoch)
    def save_img1(self, epoch):
        test_sample_num = 5
        self.G.eval()
        data_loader = self.load_data()
        for j in range(test_sample_num):
            for i, (x, y) in tqdm(enumerate(data_loader)):
                break
            x = x.to(self.device)
            y = y.to(self.device)
            content_code = self.G.encoder(x)
            fake_img1 = self.G.decoders(content_code)  # 生成假图

            image = torch.cat((x * 0.5 + 0.5, fake_img1 * 0.5 + 0.5), axis=3)
            save_image(image, os.path.join(self.result_dir, self.dataset, 'img', f"train_{j}{epoch}.png"))
        print("训练集测试图像生成成功！")
        self.save_model()
        self.G.train(), self.style_net.train(), self.sct.train()

    def save_img(self, epoch):
        test_sample_num = 1
        self.G.eval(), self.style_net.eval(), self.sct.eval()
        data_loader = self.load_data()
        for j in range(test_sample_num):
            for i, (x, y) in tqdm(enumerate(data_loader)):
                break
            x = x.to(self.device)
            y = y.to(self.device)
            rand_style = torch.randn([self.batch_size, self.hw, self.latent_dim, self.latent_dim]).to(
                self.device).requires_grad_()
            style_code = self.style_net(y)
            content_code = self.G.encoder(x)
            share_code = self.sct(content_code, style_code, rand_style)
            fake_img1 = self.G.decoders(share_code)  # 生成假图
            fake_img2 = self.gf.guided_filter(x, fake_img1, r=1)
            image = torch.cat((x * 0.5 + 0.5, fake_img1 * 0.5 + 0.5, fake_img2 * 0.5 + 0.5), axis=3)
            save_image(image, os.path.join(self.result_dir, self.dataset, 'img', f"train_{j}{epoch}.png"))
        print("训练集测试图像生成成功！")
        test_loader = self.load_data(epoch_test=True)
        for i, (x, y) in tqdm(enumerate(test_loader)):
            break
        x = x.to(self.device)
        y = y.to(self.device)
        #         style1_code,_ = self.style_net(lr_test)
        rand_style = torch.randn([4, self.hw, self.latent_dim, self.latent_dim]).to(self.device).requires_grad_()
        style_code = self.style_net(y)
        content_code = self.G.encoder(x)
        share_code = self.sct(content_code, style_code, rand_style)
        fake_img = self.G.decoders(share_code)
        lr_img2 = self.gf.guided_filter(x, fake_img, r=1)
        image = torch.cat((x * 0.5 + 0.5, fake_img * 0.5 + 0.5, lr_img2 * 0.5 + 0.5), axis=3)
        save_image(image, os.path.join(self.result_dir, self.dataset, 'img', f"test_{epoch}.png"))
        print("高分辨率测试集测试图像生成成功！")
        self.save_model()
        self.G.train(), self.style_net.train(), self.sct.train()

    # 保存模型
    def save_model(self):
        params = {}
        params["G"] = self.G.state_dict()
        params['sct'] = self.sct.state_dict()
        params["style"] = self.style_net.state_dict()
        params["D"] = self.D.state_dict()
        params["D_patch"] = self.D_patch.state_dict()
        params["iter"]=self.iter
        torch.save(params, os.path.join(self.result_dir, self.dataset, self.checkpoint_dir,
                                        f'checkpoint_{self.dataset}.pth'))
        print("保存模型成功！")

    # 加载模型
    def load_model(self):
        params = torch.load(self.test_dir)
        self.G.load_state_dict(params['G'])
        self.sct.load_state_dict(params['sct'])
        self.style_net.load_state_dict(params['style'])
        self.D.load_state_dict(params['D'])
        self.D_patch.load_state_dict(params['D_patch'])
        if params.__contains__('iter'):
            self.iter = int(params['iter'])
        print("加载模型成功！")
    def test(self):
        self.load_model()
        test_sample_num = 1
        self.G.eval(), self.style_net.eval(), self.sct.eval()
        data_loader = self.load_data()
        for j in range(test_sample_num):
            for i, (x, y) in tqdm(enumerate(data_loader)):
                x = x.to(self.device)
                y = y.to(self.device)
                # rand_style = torch.randn([2,self.hw,self.latent_dim,self.latent_dim]).to(self.device).requires_grad_()
                style_code = self.style_net(y)
                content_code = self.G.encoder(x)
                share_code = self.sct(content_code, style_code)
                fake_img = self.G.decoders(share_code)  # 生成假图
                save_image(fake_img[0] * 0.5 + 0.5,
                           os.path.join(self.result_dir, self.dataset, 'img', f'anime_lr{j}to{i}.png'))
        print("训练集测试图像生成成功！")
        # self.high_test()

    def high_test(self):
        self.load_model()
        data_loader = self.load_data(epoch_test=True, high=True)
        self.G.eval(), self.style_net.eval(), self.sct.eval()
        for i, (x, y) in tqdm(enumerate(data_loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            # rand_style = torch.randn([1,self.hw,self.latent_dim,self.latent_dim]).to(self.device).requires_grad_()
            style_code = self.style_net(y)
            content_code = self.G.encoder(x)
            share_code = self.sct(content_code, style_code)
            fake_img = self.G.decoders(share_code)
            image = torch.cat((x * 0.5 + 0.5, fake_img * 0.5 + 0.5), axis=3)
            save_image(image, os.path.join(self.result_dir, self.dataset, 'img', f"test_high{i}.png"))
            print("高分辨率测试集测试图像生成成功！")
