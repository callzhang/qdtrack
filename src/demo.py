from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pandas.core import frame
from PIL import Image, ImageDraw
import _init_paths
import logging
import os, cv2
from glob import glob
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import pandas as pd
from tqdm import tqdm
import numpy as np

logger.setLevel(logging.INFO)

class_ID = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 3,
    'Cyclist': 3,
    'Tram': 4,
    'Misc': 5,
    'DontCare': 6,
    'Person': 3,
    'ignored': 7
}
ID_class = {v: k for k, v in class_ID.items()}


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = opt.result_file
    fps = dataloader.frame_rate
    opt.track_buffer = fps * opt.track_duration
    opt.fps = fps

    frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame')
    eval_seq(opt, dataloader, opt.dataset, result_filename,
             save_dir=frame_dir, show_image=opt.show_image, frame_rate=fps,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        post_processing(opt)
        # filename = opt.input_video.split('/')[-1].split('.')
        # filename = filename[0]+'_result.'+filename[-1]
        # output_video_path = os.path.join(result_root, filename)
        # cmd_str = f'ffmpeg -y -framerate {fps} -f image2 -i {os.path.join(result_root, "frame")}/%05d.jpg -c:v libx264 -preset fast -x264-params crf=25 -vf fps={fps} {output_video_path}'
        # logger.info(f'Running ffmpeg with cmd:\n{cmd_str}')
        # os.system(cmd_str)

def post_processing(opt):
    results = pd.read_csv(opt.result_file)
    temp_video = opt.result_file.replace('.txt', '_temp.mp4')
    ids = results.id.unique()
    ids.sort()
    frames = results.frame.unique()
    frames.sort()
    # set up video loading and writer
    cap = cv2.VideoCapture(opt.input_video)
    frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(temp_video, fourcc, frame_rate, size)
    # detection is frame x id x 6 (x1, y1, x2, y2, label, score) matrix
    detections = np.zeros((n_frame, len(ids), 6))
    #fill value
    for tid in tqdm(ids):
        tracks = results.query('id == @tid')
        # remove short ids
        if len(tracks) < 30:
            print(f'Skip object [{tid}] with only {len(tracks)} frames')
            continue
        # get frame range for track id
        fs = tracks.frame.to_numpy()
        f_range = range(fs.min(), fs.max())
        # fill value
        id_ind = np.where(ids==tid)
        detections[fs, id_ind, 4] = tracks.label
        detections[fs, id_ind, 5] = tracks.score
        # interpolate between frames
        for i, yi in enumerate(['x1', 'y1', 'x2', 'y2']):
            yp = tracks[yi]
            y_interp = np.interp(f_range, fs, yp)
            detections[f_range, id_ind, i] = y_interp

    # render video
    colors = np.random.randint(1, 255, (len(ids), 3))
    for frame in tqdm(range(n_frame), desc=f'Rendering {opt.output_video_path}'):
        res, img0 = cap.read()  # BGR
        assert res is True
        img = Image.fromarray(img0)
        draw = ImageDraw.Draw(img)
        frame_data = detections[frame]
        for ix, track_data in enumerate(frame_data):
            tid = ids[ix]
            x1, y1, x2, y2, label, score = track_data
            if track_data.sum() == 0:
                continue
            c = tuple(colors[ix])
            draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
            label_str = ID_class[label-1] if (label-1) in ID_class else "Unknown"
            draw.text((x1, y1), f'{label_str}:{score:.2f}%({tid})', fill=c)
        img1 = np.asarray(img)
        videoWriter.write(img1)
    videoWriter.release()
    # convert to h264
    filename = opt.input_video.split('/')[-1].split('.')
    filename = filename[0]+'_result.'+filename[-1]
    output_video_path = os.path.join(opt.output_root, filename)
    cmd_str = f'ffmpeg -y -i {temp_video} -vcodec libx264 -c:v libx264 -preset fast -x264-params crf=25 -vf fps={frame_rate} {output_video_path}'
    os.system(cmd_str)
    os.remove(temp_video)
    print(f'finished, output saved to {output_video_path}')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init(
        [
            # '--load_model', 'models/ctdet_coco_dla_2x.pth',
            '--load_model', 'models/fairmot_dla34.pth',
            '--input_video', 'videos/video.mp4',
            '--output_root', 'output',
            '--num_classes', '1',
            # '--ltrb', False,
            '--dataset', 'kitti'
    ])
    demo(opt)

    #kitti
    # for v in glob('videos/kitti/*.mp4'):
    #     opt.input_video = v
    #     demo(opt)

    #bosch
    # opt.output_root = 'output/bosch'
    # for v in glob('videos/bosch_tracking/*.mp4')[1:]:
    #     opt.input_video = v
    #     demo(opt)
