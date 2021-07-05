import pandas as pd
import numpy as np
import os, cv2, pickle
from tqdm import tqdm
from PIL import Image, ImageDraw


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


def post_processing(result_file, input_video, ID_class, output_video_path = None):
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
        if len(tracks) < frame_rate:
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
            y_smooth = np.convolve(y_interp, frame_rate, mode='same')
            assert len(y_smooth) == len(y_interp)
            from matplotlib import pyplot as pt
            pt.plot(f_range, y_smooth)
            detections[f_range, id_ind, i] = y_smooth

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
            label = int(label)
            if track_data.sum() == 0:
                continue
            c = tuple(colors[ix])
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
    print('finished')


if __name__ == '__main__':
    result = pickle.load(open('temp/result', 'rb'))
    output = convert_result(result)
