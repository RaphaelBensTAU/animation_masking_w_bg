import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np
from sync_batchnorm import DataParallelWithCallback
from skimage import io, img_as_float32
import torch.nn.functional as F

def binarize_mask(mask, thresh):
        mask = (mask * 255)
        mask[mask > thresh] = 255
        mask[mask <= thresh] = 0
        mask /= 255
        return mask
        
def segment_using_mask(fg_mask, img, thresh):
        mask = binarize_mask(fg_mask, thresh)
        foreground = img * mask
        background = img * (1 - mask)
        return foreground, background, mask


def animate(config, generator, mask_generator,bg_base, bg_refine, checkpoint, log_dir, dataset, third_image_not_from_dataset = False):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, mask_generator=mask_generator, bg_generator = bg_base, bg_hd = bg_refine)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        mask_generator = DataParallelWithCallback(mask_generator)

    generator.eval()
    mask_generator.eval()
    bg_base.eval()
    bg_refine.eval()

    c = 0

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            source_mask = mask_generator(source_frame)
            
            third_img = torch.from_numpy(img_as_float32(io.imread('bulgaria.jpg')).transpose(2, 0, 1)).cuda()[None,:,:,:]
            downsampled_third_img = F.interpolate(third_img, size=(64, 64), mode='bilinear')

            if not third_image_not_from_dataset:
                masked_third = mask_generator(third_img)
                masked_third = F.pad(input=masked_third, pad=(3, 3, 3, 3), mode='constant', value=0)
                mask_third_upsampled = F.interpolate(masked_third, size=(256, 256), mode='bilinear')

            for frame_idx in range(driving_video.shape[2]):
                print(c)
                c += 1
                driving_frame = driving_video[:, :, frame_idx]
                driving_mask = mask_generator(driving_frame)

                out = generator(source_frame, driving_frame, mask_source=source_mask, mask_driving=driving_mask, animate=True, mask_driving2=None, predict_mask=False)

                downsampled_second_phase = F.interpolate(out['second_phase_prediction'], size=(64, 64), mode='bilinear')

                if third_image_not_from_dataset:
                    masked_third = out['fixed_mask_64']
                    mask_third_upsampled = out['fixed_mask']

                foreground_src, background_src, binary_src_mask = segment_using_mask(masked_third, downsampled_third_img, 60)
                foreground_driving, background_driving, binary_driving_mask = segment_using_mask(out['fixed_mask_64'], downsampled_second_phase, 60)

                input_ = torch.cat((binary_src_mask.detach(), binary_driving_mask.detach()), dim=1)
                input_ = torch.cat((input_, background_src.detach()), dim=1)
                input_ = torch.cat((input_, foreground_driving.detach()), dim=1)
                generated_bg = bg_base(input_)


                foreground_src_hd, background_src_hd, binary_src_mask_hd = segment_using_mask(mask_third_upsampled, third_img, 60)
                foreground_driving_hd, background_driving_hd, binary_driving_mask_hd = segment_using_mask(out['fixed_mask'], out['second_phase_prediction'], 60)

                input_ = torch.cat((binary_src_mask_hd.detach(), binary_driving_mask_hd.detach()), dim=1)
                input_ = torch.cat((input_, background_src_hd.detach()), dim=1)
                input_ = torch.cat((input_, foreground_driving_hd.detach()), dim=1)
                input_ = torch.cat((input_, generated_bg.detach()), dim=1)
                generated_bg_hd = bg_refine(input_)
                
                #pyramide_bg_hd = self.pyramid(generated_bg_hd)

    
            
                # out['driving_mask'] = driving_mask
                # out['source_mask'] = source_mask
                
    
                predictions.append(np.transpose(out['first_phase_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                predictions.append(np.transpose(out['second_phase_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                predictions.append(np.transpose(out['second_phase_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize_with_bg(source_frame, driving_frame, None, out['second_phase_prediction'],third_img,generated_bg, generated_bg_hd )
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            #imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
