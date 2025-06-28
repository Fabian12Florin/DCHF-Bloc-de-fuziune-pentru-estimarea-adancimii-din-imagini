import argparse
import os
import sys
import time

from PIL import Image, ImageFile
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage

from models.get_models import get_model_with_opts
from saver import load_model_for_evaluate
from utils.platform_loader import read_yaml_options
from visualizer import Visualizer

sys.path.append(os.getcwd())
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='SMDE Prediction Parser')

parser.add_argument('--image_path',
                    dest='image_path',
                    required=True,
                    help='the path to the images')
parser.add_argument('--exp_opts',
                    dest='exp_opts',
                    required=True,
                    help="the yaml file for model's options")
parser.add_argument('--model_path',
                    dest='trained_model',
                    required=True,
                    help='the path of trained model')
parser.add_argument('--input_size',
                    dest='input_size',
                    type=int,
                    nargs='+',
                    default=None,
                    help='the size of input images')
parser.add_argument('--out_dir',
                    dest='out_dir',
                    type=str,
                    default=None,
                    help='the folder name for the outputs')

parser.add_argument('--cpu',
                    dest='cpu',
                    action='store_true',
                    default=False,
                    help='predicting with cpu')
parser.add_argument('-gpp',
                    '--godard_post_process',
                    action='store_true',
                    default=False,
                    help='Post-processing as done in Godards paper')
parser.add_argument('-mspp',
                    '--multi_scale_post_process',
                    action='store_true',
                    default=False,
                    help='Post-processing as done in FAL-Net')
parser.add_argument('--show_grid',
                    dest='show_grid',
                    action='store_true',
                    default=True,
                    help='Show advanced grid visualization after prediction')
parser.add_argument('--save_grid',
                    dest='save_grid',
                    action='store_true',
                    default=True,
                    help='Save grid visualization as PNG files')

opts = parser.parse_args()


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in
    Monodepthv1."""
    _, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    l_mask = torch.from_numpy(l_mask.copy()).unsqueeze(0).to(l_disp)
    r_mask = torch.from_numpy(r_mask.copy()).unsqueeze(0).to(l_disp)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def multi_scale_post_process(l_disp, r_down_disp):
    norm = l_disp / (np.percentile(l_disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * l_disp + norm * r_down_disp

def create_advanced_depth_grid(original_img, depth_raw, disparity_raw, base_name, save_path,
                               show_plot=True, save_plot=True):
    print(f"Creez grila de analiza pentru: {base_name}")
    
    disp_norm = (disparity_raw - disparity_raw.min()) / (disparity_raw.max() - disparity_raw.min() + 1e-8)

    zones = np.zeros_like(depth_raw)
    p10, p25, p40, p60, p75, p90 = np.percentile(depth_raw, [10, 25, 40, 60, 75, 90])
    zones[depth_raw <= p10] = 1
    zones[(depth_raw > p10) & (depth_raw <= p25)] = 2
    zones[(depth_raw > p25) & (depth_raw <= p40)] = 3
    zones[(depth_raw > p40) & (depth_raw <= p60)] = 4
    zones[(depth_raw > p60) & (depth_raw <= p75)] = 5
    zones[(depth_raw > p75) & (depth_raw <= p90)] = 6
    zones[depth_raw > p90] = 7

    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(original_img)
    ax1.set_title(f'IMAGINEA ORIGINALA', fontsize=14, fontweight='bold', color='darkblue')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(disp_norm, cmap='plasma')
    ax2.set_title(f'DISPARITATE', fontsize=14, fontweight='bold', color='darkred')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(zones, cmap='Spectral_r', interpolation='nearest')
    ax3.set_title(f'ZONE DE ADANCIME', fontsize=14, fontweight='bold', color='darkgreen')
    ax3.axis('off')
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(1, 8))
    cbar.set_ticklabels(['Nivel 1', 'Nivel 2', 'Nivel 3', 'Nivel 4', 'Nivel 5', 'Nivel 6', 'Nivel 7'])

    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')

    zone_pcts = [np.sum(zones == i) / zones.size * 100 for i in range(1, 8)]

    stats_text = f'''
ANALIZA {base_name}

DISTRIBUTIA PE ZONE:
• Zona 1: {zone_pcts[0]:5.1f}%
• Zona 2: {zone_pcts[1]:5.1f}%
• Zona 3: {zone_pcts[2]:5.1f}%
• Zona 4: {zone_pcts[3]:5.1f}%
• Zona 5: {zone_pcts[4]:5.1f}%
• Zona 6: {zone_pcts[5]:5.1f}%
• Zona 7: {zone_pcts[6]:5.1f}%

REZOLUTIA IMAGINII:
• {depth_raw.shape[1]} x {depth_raw.shape[0]} pixeli
    '''

    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', alpha=0.9, 
                      edgecolor='darkblue', linewidth=1))

    plt.suptitle(f'ANALIZA pentru imaginea: {base_name}', fontsize=16, fontweight='bold', y=0.98, color='navy')
    plt.tight_layout()

    if save_plot:
        grid_path = os.path.join(save_path, f"{base_name}_analiza.png")
        plt.savefig(grid_path, dpi=200, bbox_inches='tight')
        print(f"Grila salvata: {grid_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
def predict_one_image(network, inputs, visualizer, save_path, file, original_img):
    outputs = {}
    network(inputs['color_s'], outputs, is_train=False)

    if opts.godard_post_process:
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        flip_outputs = {}
        network(inputs['color_s'], flip_outputs, is_train=False)
        fflip_depth = torch.flip(flip_outputs[('depth', 's')], dims=[3])
        pp_depth = batch_post_process_disparity(1 / outputs[('depth', 's')], 1 / fflip_depth)
        pp_depth = 1 / pp_depth
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        outputs[('depth', 's')] = pp_depth
    elif opts.multi_scale_post_process:
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        up_fac = 2/3
        H, W = inputs['color_s'].shape[2:]
        inputs['color_s'] = F.interpolate(inputs['color_s'], scale_factor=up_fac, mode='bilinear', align_corners=True)
        flip_outputs = {}
        network(inputs['color_s'], flip_outputs, is_train=False)
        flip_depth = flip_outputs[('depth', 's')]
        flip_depth = up_fac * F.interpolate(flip_depth, size=(H, W), mode='nearest')
        fflip_depth = torch.flip(flip_depth, dims=[3])
        pp_depth = batch_post_process_disparity(1 / outputs[('depth', 's')], 1 / fflip_depth)
        pp_depth = 1 / pp_depth
        inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
        outputs[('depth', 's')] = pp_depth
    else:
        pp_depth = outputs[('depth', 's')]



    
    # Extragere depth rafinat
    if 'refmono_depth_0_s' in outputs:
        refined_depth_np = outputs['refmono_depth_0_s'].squeeze(0).squeeze(0).cpu().numpy()

    # Extragere disparitate din outputs
    if 'refmono_disp_0_s' in outputs:
        disparity_np = outputs['refmono_disp_0_s'].squeeze(0).squeeze(0).cpu().numpy()

    base_name = os.path.splitext(file)[0]
    np.save(os.path.join(save_path, base_name + '_pred.npy'), pp_depth.squeeze(0).squeeze(0).cpu().numpy())
    if 'refmono_depth_0_s' in outputs or 'refdepth' in outputs:
        np.save(os.path.join(save_path, base_name + '_refined.npy'), refined_depth_np)

    if opts.show_grid or opts.save_grid:
        create_advanced_depth_grid(
            original_img=original_img,
            depth_raw=refined_depth_np,
            disparity_raw=disparity_np,
            base_name=base_name,
            save_path=save_path,
            show_plot=opts.show_grid,
            save_plot=opts.save_grid
        )

def predict():
    # Inițializarea dispozitivului (CPU/GPU)
    if opts.cpu:
        device = torch.device('cpu')
        print('Folosesc CPU pentru predictii')
    else:
        device = torch.device('cuda')
        print('Folosesc GPU pentru predictii')

    opts_dic = read_yaml_options(opts.exp_opts)

    print('Incarc modelul pre-antrenat...')
    print('Calea catre model: {}'.format(opts.trained_model))
    network = get_model_with_opts(opts_dic, device)
    network = load_model_for_evaluate(opts.trained_model, network)
    network.eval()
    print('Modelul este gata pentru predictii!')

    if opts.out_dir is not None:
        os.makedirs(opts.out_dir, exist_ok=True)
        save_path = opts.out_dir
    else:
        if os.path.isfile(opts.image_path):
            save_path = os.path.dirname(opts.image_path)
        else:
            save_path = opts.image_path
    
    #visualizer = Visualizer(save_path, {'type':{'pp_depth': 'depth'},
    #                                    'shape': [['pp_depth']]})

    visualizer = Visualizer(save_path, {'type':{'refmono_disp_0_s': 'disp', 'color_s':'img'},
                                        'shape': [['color_s'], ['refmono_disp_0_s']]})

    to_tensor = tf.ToTensor()
    normalize = tf.Normalize(mean=opts_dic['pred_norm'],
                             std=[1, 1, 1])
    if opts.input_size is not None:
        image_size = opts.input_size
    else:
        image_size = opts_dic['pred_size']
    print('Redimensionez imaginea/imaginile la: {}'.format(image_size))
    resize = tf.Resize(image_size,interpolation=Image.LANCZOS)

    if opts.godard_post_process or opts.multi_scale_post_process:
        print('Folosesc post-procesare avansata pentru rezultate mai bune')
    
    print('Incep predictiile...')
    print(f'Vizualizare grid: {"ACTIVATA" if opts.show_grid else "DEZACTIVATA"}')
    print(f'Salvare grid-uri: {"ACTIVATA" if opts.save_grid else "DEZACTIVATA"}')
    
    start_time = time.time()
    processed_count = 0
    
    with torch.no_grad():
        if os.path.isfile(opts.image_path):
            print(f"Procesez o singura imagine: {opts.image_path}")
            img = Image.open(opts.image_path)
            img = img.convert('RGB')
            original_img = np.array(resize(img))
            img_tensor = normalize(to_tensor(resize(img))).unsqueeze(0)
            inputs = {}
            inputs['color_s'] = img_tensor.to(device)
            file = os.path.basename(opts.image_path)
            predict_one_image(network, inputs, visualizer, save_path, file, original_img)
            processed_count = 1
        else:
            print(f"Procesez dosarul: {opts.image_path}")
            supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            for r, ds, fs in os.walk(opts.image_path):
                for f in fs:
                    if any(f.lower().endswith(fmt) for fmt in supported_formats):
                        try:
                            print(f"Procesez: {f}")
                            img = Image.open(os.path.join(r, f))
                            img = img.convert('RGB')
                            original_img = np.array(resize(img))
                            img_tensor = normalize(to_tensor(resize(img))).unsqueeze(0)
                            inputs = {}
                            inputs['color_s'] = img_tensor.to(device)
                            predict_one_image(network, inputs, visualizer,
                                              save_path, f, original_img)
                            processed_count += 1
                        except Exception as e:
                            print(f"Eroare la procesarea {f}: {str(e)}")
                            continue

    duration = time.time() - start_time
    print(f'Predictii finalizate cu succes!')
    print(f'Am procesat {processed_count} imagini in {duration:.1f} secunde')
    print(f'Fisierele de predictie sunt salvate in: {save_path}')
    if opts.save_grid:
        print(f'Grid-urile de analiza sunt salvate ca *_analiza.png')

if __name__ == '__main__':
    predict()
