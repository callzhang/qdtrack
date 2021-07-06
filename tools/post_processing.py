import pandas as pd
import numpy as np
import os, cv2, pickle
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from simplification.cutil import simplify_coords_idx
from collections import defaultdict


def video_to_images(video_path, image_folder=None):
    if image_folder is None:
        image_folder = 'temp/' + video_path.split('/')[-1].split('.')[0]
    os.makedirs(image_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f = 0
    for i in tqdm(range(n), desc='Export image'):
        res, img = cap.read()
        if not res:
            break
        path = f'{image_folder}/{f:05d}.jpg'
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        f +=1
    return image_folder


def convert_result(result, output='temp/result.csv'):
    result = result['track_result']
    data = []
    for f, tracks in enumerate(result):
        for id, bbox in tracks.items():
            x1, y1, x2, y2, conf = bbox['bbox']
            label = bbox['label']
            data.append({
                'frame': f,
                'label': label,
                'id': id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'score': conf
            })
    data = pd.DataFrame(data)
    data.to_csv(output)
    return output


def post_processing(result_file, input_video, ID_class, output_video_path = None, get_video = False):
    results = pd.read_csv(result_file)
    temp_video = result_file.replace('.csv', '_temp.mp4')
    if not output_video_path:
        output_video_path = f"output/{input_video.split('/')[-1].replace('.mp4', '_result.mp4')}"
    ids = results.id.unique()
    ids.sort()
    frames = results.frame.unique()
    frames.sort()
    # set up video loading and writer
    cap = cv2.VideoCapture(input_video)
    frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    ws = int(frame_rate/3)+1
    ws = ws if ws % 2 == 1 else ws - 1
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(temp_video, fourcc, frame_rate, size)
    # detection is frame x id x 6 (x1, y1, x2, y2, label, score) matrix, default is -1
    detections = np.full((n_frame, len(ids), 6), -1, dtype=np.float32)
    #fill value
    for tid in tqdm(ids, desc='interpolating'):
        tracks = results.query('id == @tid')
        # remove short ids
        if len(tracks) < frame_rate:
            print(f'Skip object [{tid}] with only {len(tracks)} frames')
            continue
        # get frame range for track id
        fs = tracks.frame.to_numpy()
        f_range = np.array(range(fs.min(), fs.max()))
        # fill value
        id_ind = np.where(ids==tid)
        detections[fs, id_ind, 4] = tracks.label
        detections[fs, id_ind, 5] = tracks.score
        # interpolate between frames
        for i, yi in enumerate(['x1', 'y1', 'x2', 'y2']):
            yp = tracks[yi]
            y_interp = np.interp(f_range, fs, yp)
            # y_smooth = np.convolve(y_interp, np.ones(frame_rate)/frame_rate, mode='same')
            y_smooth = savgol_filter(y_interp, ws, 3)
            assert len(y_smooth) == len(y_interp)
            # from matplotlib import pyplot as pt
            # pt.plot(f_range, y_smooth)
            detections[f_range, id_ind, i] = y_smooth
        #simplify
        #TODO: use better simplification algorithm
        detections_origin = detections.copy() # for debugging
        x1y1 = detections[f_range, id_ind, slice(0,3,2)].squeeze(0)
        x2y2 = detections[f_range, id_ind, slice(1,4,2)].squeeze(0)
        simp_ind_x1y1 = simplify_coords_idx(x1y1, 1.0)
        simp_ind_x2y2 = simplify_coords_idx(x2y2, 1.0)
        simp_ind = set(simp_ind_x1y1) | set(simp_ind_x2y2)
        simp_ind = np.array(sorted(list(simp_ind)), dtype=int)
        simp_ind = f_range[simp_ind]
        simp_coord = detections[simp_ind, id_ind, 0:4].copy()
        detections[f_range, id_ind, 0:4] = -1
        detections[simp_ind, id_ind, 0:4] = simp_coord
        if get_video: # interpolate back from simplified for video rendering
            for i in range(4):
                y = detections[simp_ind, id_ind, i].squeeze(0)
                detections[f_range, id_ind, i] = np.interp(f_range, simp_ind, y)

    if not get_video:
        return detections

    # render video
    colors = np.random.randint(1, 255, (len(ids), 3))
    for frame in tqdm(range(n_frame), desc=f'Rendering {output_video_path}'):
        res, img0 = cap.read()  # BGR
        assert res is True
        img = Image.fromarray(img0)
        draw = ImageDraw.Draw(img)
        frame_data = detections[frame]
        for ix, track_data in enumerate(frame_data):
            tid = ids[ix]
            x1, y1, x2, y2, label, score = track_data
            if sum([x1, y1, x2, y2]) <= 0:
                continue
            c = tuple(colors[ix])
            label = int(label)
            draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
            if isinstance(ID_class, (tuple, list)):
                label_str = ID_class[int(label)]
            elif isinstance(ID_class, dict):
                label_str = ID_class[label-1]# if (label-1) in ID_class else "Unknown"
            else:
                raise Exception('Unknown ID_class type')
            draw.text((x1, y1), f'{label_str}:{score:.2f}%({tid})', fill=c)
        img1 = np.asarray(img)
        videoWriter.write(img1)
    videoWriter.release()
    # convert to h264
    filename = input_video.split('/')[-1].split('.')
    filename = filename[0]+'_result.'+filename[-1]
    cmd_str = f'ffmpeg -y -i {temp_video} -vcodec libx264 -c:v libx264 -preset fast -x264-params crf=25 -vf fps={frame_rate} {output_video_path}'
    os.system(cmd_str)
    os.remove(temp_video)
    print(f'finished {output_video_path}')
    return output_video_path


if __name__ == '__main__':
    result = pickle.load(open('temp/result', 'rb'))
    output = convert_result(result)
