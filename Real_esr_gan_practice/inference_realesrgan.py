import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# input = input 이미지 폴더 명
# outscale = upscaling
# suffix = output 저장 이름
# model_name = 모델 파일
# output = output이 저장될 폴더 명
# tile = tile size
# ext : 이미지 확장자 명
def real_esrgan(input = 'inputs', model_name='RealESRGAN_x4plus', output = 'results', outscale = 4, 
         suffix = 'out', tile = 0, tile_pad = 10,pre_pad = 0, alpha_upsampler='realesrgan', ext = 'auto',face_enhance=True,fp32=True):
    """Inference demo for Real-ESRGAN.
    """

    # determine model paths
    # 모델 불러오기
    model_path = os.path.join(model_name+ '.pth')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    
    # 모델이 없는 경우
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32)

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(output, exist_ok=True)

    if os.path.isfile(input):
        paths = [input]
    else:
        paths = sorted(glob.glob(os.path.join(input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        # 이미지 로드
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        # 모델 실행
        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if suffix == '':
                save_path = os.path.join('results', f'{imgname}.{extension}')
            else:
                save_path = os.path.join('results', f'{imgname}_{suffix}.{extension}')
                
            # 결과 저장
            cv2.imwrite(save_path, output)