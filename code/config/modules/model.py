from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d
from torchvision import models
import numpy as np
from torch.autograd import grad

import torchvision
from torchvision.utils import save_image


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class GeneratorFullModel(torch.nn.Module):
    def __init__(self, mask_generator, generator, bg_generator,bg_refiner, train_params):
        super(GeneratorFullModel, self).__init__()
        self.mask_generator = mask_generator
        self.generator = generator
        self.bg_generator = bg_generator
        self.bg_refiner = bg_refiner
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.c = 0

        self.loss_weights = train_params['loss_weights']
        self.l1 = nn.L1Loss()

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def binarize_mask(self, mask, thresh):
        mask = (mask * 255)
        mask[mask > thresh] = 255
        mask[mask <= thresh] = 0
        mask /= 255
        return mask
        
    def segment_using_mask(self, fg_mask, img, thresh):
        mask = self.binarize_mask(fg_mask, thresh)
        foreground = img * mask
        background = img * (1 - mask)
        return foreground, background, mask

    def forward(self, x, predict_mask, predict_bg):
        mask_source = self.mask_generator(x['source'])
        mask_driving = self.mask_generator(x['driving'])
        mask_driving2 = self.mask_generator(x['driving2'])

        generated = self.generator(x['source'], x['driving'], mask_source=mask_source, mask_driving=mask_driving, mask_driving2=mask_driving2, predict_mask=predict_mask, animate=False)
        generated.update({'r_mask_source': mask_source, 'r_mask_driving': mask_driving, 'r_mask_driving2': mask_driving2})

        if predict_bg:
            with torch.no_grad():
                downsampled_target = F.interpolate(x['target'],size=(64,64), mode='bilinear')
                foreground_src, background_src, binary_src_mask = self.segment_using_mask(generated['source_mask_64'], generated['source_64'], 60)
                foreground_driving, background_driving, binary_driving_mask = self.segment_using_mask(generated['driving_mask_64'],downsampled_target, 60)

            input_ = torch.cat((binary_src_mask.detach(), binary_driving_mask.detach()), dim=1)
            input_ = torch.cat((input_, background_src.detach()), dim=1)
            input_ = torch.cat((input_, foreground_driving.detach()), dim=1)
            generated_bg = self.bg_generator(input_)
            pyramide_bg = self.pyramid(generated_bg)
            generated['generated_bg'] = generated_bg

            with torch.no_grad():
                foreground_src_hd, background_src_hd, binary_src_mask_hd = self.segment_using_mask(generated['source_mask_256'], x['source'], 60)
                foreground_driving_hd, background_driving_hd, binary_driving_mask_hd = self.segment_using_mask(generated['driving_mask_256'],x['target'], 60)

            input_ = torch.cat((binary_src_mask_hd.detach(), binary_driving_mask_hd.detach()), dim=1)
            input_ = torch.cat((input_, background_src_hd.detach()), dim=1)
            input_ = torch.cat((input_, foreground_driving_hd.detach()), dim=1)
            input_ = torch.cat((input_, generated_bg.detach()), dim=1)
            generated_bg_hd = self.bg_refiner(input_)
            pyramide_bg_hd = self.pyramid(generated_bg_hd)
            generated['generated_bg_hd'] = generated_bg_hd
            
            with torch.no_grad():
                y = torch.zeros([6, 3, 256, 256], dtype=torch.float32)
                y[0] = x['source'][0]
                y[1] = foreground_driving_hd[0]
                y[2] = background_src_hd[0]
                y[3] = x['target'][0]
                y[4] = generated_bg[0]
                y[5] = generated_bg_hd[0]
                save_image(y, '/content/drive/My Drive/DL_Project_1/Code/log_bg/folder/{}.png'.format(self.c))
                #save_image(generated_bg[0], '/content/drive/My Drive/DL_Project_1/Code/log_bg/folder/{}_2.png'.format(self.c))
                self.c += 1

        loss_values = {}
        bg_losses = {}

        if predict_mask:
            loss_values['mask_correction'] = 100 * self.l1(generated['fixed_mask'] ,generated['driving_mask_int_detached'])
        else:
            loss_values['mask_correction'] = torch.zeros(10).cuda()

        pyramide_real = self.pyramid(x['target'])
        pyramide_first_generated = self.pyramid(generated['first_phase_prediction'])
        pyramide_second_generated = self.pyramid(generated['second_phase_prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total1 = 0
            value_total2 = 0

            if predict_bg:
                value_total3 = 0
                value_total4 = 0

            for scale in self.scales:
                x_vgg1 = self.vgg(pyramide_first_generated['prediction_' + str(scale)])
                x_vgg2 = self.vgg(pyramide_second_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                if predict_bg:
                    bg_vgg = self.vgg(pyramide_bg['prediction_' + str(scale)])
                    bg_vgg_hd = self.vgg(pyramide_bg_hd['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value1 = torch.abs(x_vgg1[i] - y_vgg[i].detach()).mean()
                    value2 = torch.abs(x_vgg2[i] - y_vgg[i].detach()).mean()

                    value_total1 += self.loss_weights['perceptual'][i] * value1
                    value_total2 += self.loss_weights['perceptual'][i] * value2

                    if predict_bg:
                        value_total3 += self.loss_weights['perceptual'][i] * torch.abs(bg_vgg[i] - y_vgg[i].detach()).mean()
                        value_total4 += self.loss_weights['perceptual'][i] * torch.abs(bg_vgg_hd[i] - y_vgg[i].detach()).mean()
            loss_values['perceptual1'] = value_total1
            loss_values['perceptual2'] = value_total2

            if predict_bg:
                bg_losses['perceptual_bg'] = value_total3
                bg_losses['perceptual_bg_hd'] = value_total4

        return loss_values, bg_losses, generated
        #return loss_values, generated
