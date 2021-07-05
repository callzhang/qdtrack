import warnings
from matplotlib.patches import Rectangle
from matplotlib.pyplot import draw
# from bdd100k.label.to_coco import init

import mmcv, cv2, torch
from numpy.lib.arraysetops import isin
import numpy as np
import os, pickle
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from qdtrack.datasets.pipelines.formatting import VideoCollect
from qdtrack.models import build_model
from glob import glob
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm

def init_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    # config.model.pretrained = None
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.model.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    wrap_fp16_model(model)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files, vedio path or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    def video_generator(cap):
        while True:
            res, img = cap.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img

    if isinstance(imgs, (list, tuple)):
        is_batch = True
        n_frame = len(imgs)
        pipeline_cfg = cfg.data.test.pipeline
    elif isinstance(imgs, str) and imgs.endswith('.mp4'):
        is_batch = True
        cap = cv2.VideoCapture(imgs)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        imgs = video_generator(cap)
        pipeline_cfg = cfg.data.video.pipeline
    else:
        raise Exception('does not support single image')

    # cfg.data.inference.pipeline = replace_ImageToTensor(cfg.data.inference.pipeline)
    # pipeline = Compose(cfg.data.inference.pipeline)
    cfg.data.test.pipeline = replace_ImageToTensor(pipeline_cfg)
    pipeline = Compose(cfg.data.test.pipeline)

    results = defaultdict(list)
    with torch.no_grad():
        for i, img in tqdm(enumerate(imgs), total=n_frame, desc='tracking'):
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img, img_prefix=None, frame_id=i,
                            img_info=dict(filename=f'{img}_{i}.jpg'))
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline

            data = pipeline(data)
            data['img_metas'] = [[data['img_metas'][0].data]]
            data['img_metas'][0][0]['frame_id'] = i
            data['img'] = [data['img'][0].data.unsqueeze(0).cuda()]

            # forward the model 
            result = model(return_loss=False, rescale=True, **data)

            for k, v in result.items():
                results[k].append(v)

        # results = model(return_loss=False, rescale=True, detection_only=True, **data)
    results = dict(results)
    if not is_batch:
        return results[0]
    else:
        return results


def export_video(imgs, results, classes, filename=None):
    os.makedirs('temp', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    temp_video = 'temp/tmp.mp4'
    frame_rate = 10
    if isinstance(imgs, list):
        if filename:
            output_video_path = 'output/' + filename
        else:
            output_video_path = 'output/' + imgs[0].split('/')[-2] + '.mp4'
        if isinstance(imgs[0], str):
            #list of images
            img = Image.open(imgs[0])
            n_frame = len(imgs)
            size = img.size
        elif isinstance(imgs[0], np.ndarray):
            n_frame = len(imgs)
            size = imgs[0].shape[2:]
    elif isinstance(imgs, str) and imgs.endswith('.mp4'):
        output_video_path = imgs
        cap = cv2.VideoCapture(imgs)
        frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        raise Exception('unsupported imgs format')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(temp_video, fourcc, frame_rate, size)

    #draw
    assert len(imgs) == len(results['track_result'])
    for f, (path, track_result) in tqdm(enumerate(zip(imgs, results['track_result'])), desc='rendering', total=n_frame):
        img0 = cv2.imread(path)
        img = Image.fromarray(img0)
        draw = ImageDraw.Draw(img)
        for tid, bbox in track_result.items():
            xyxy=list(bbox['bbox'][:4])
            conf = bbox['bbox'][-1]
            label = bbox['label']
            name = classes[label]
            draw.rectangle(xyxy, outline=(0,255,0), width=2)
            draw.text([xyxy[0], xyxy[1]], f'{name}|{tid}|{conf*100:.1f}%')
        rendered = np.asarray(img)
        videoWriter.write(rendered)
    videoWriter.release()
    # convert to h264
    cmd_str = f'ffmpeg -y -i {temp_video} -vcodec libx264 -c:v libx264 -preset fast -x264-params crf=25 -vf fps={frame_rate} {output_video_path}'
    os.system(cmd_str)
    os.remove(temp_video)
    print('finished')


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       title='result',
                       block=True,
                       wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI. Default: True
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    warnings.warn('"block" will be deprecated in v2.9.0,'
                  'Please use "wait_time"')
    warnings.warn('"fig_size" are deprecated and takes no effect.')
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

if __name__ == '__main__':
    model = init_model(
        config='configs/qdtrack-basch.py',
        checkpoint='models/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth',
    )
    imgs = glob('data/DETRAC/train/images/MVI_20012/*.jpg')
    # load video
    # imgs = []
    # cap = cv2.VideoCapture('data/DETRAC/MVI_20064.mp4')
    # while True:
    #     res, img = cap.read()  # BGR
    #     if not res:
    #         break
    #     img = img[:, :, ::-1].transpose(2, 0, 1) #RGB
    #     imgs.append(img)
    
    result = inference_model(model, imgs)
    pickle.dump(result, open('temp/result', 'wb'))
    result = pickle.load(open('temp/result', 'rb'))
    export_video(imgs, result, model.CLASSES)
