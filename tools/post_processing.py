import pandas as pd
import numpy as np
import os, cv2, pickle
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from simplification.cutil import simplify_coords_idx


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range]
                for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


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
    # detection is frame x id x 6 (x1, y1, x2, y2, label, score) matrix
    detections = np.zeros((n_frame, len(ids), 6))
    detections_simplified = np.zeros((n_frame, len(ids), 6))
    #fill value
    for tid in tqdm(ids, desc='interpolating'):
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
            # y_smooth = np.convolve(y_interp, np.ones(frame_rate)/frame_rate, mode='same')
            y_smooth = savgol_filter(y_interp, ws, 3)
            assert len(y_smooth) == len(y_interp)
            # from matplotlib import pyplot as pt
            # pt.plot(f_range, y_smooth)
            detections[f_range, id_ind, i] = y_smooth
        #simplify
        detections_simplified[f_range, id_ind, 4] = tracks.label[0]
        detections_simplified[fs, id_ind, 5] = tracks.score
        x1y1 = detections[f_range, id_ind, slice(0,3,2)].squeeze(0)
        x2y2 = detections[f_range, id_ind, slice(1,4,2)].squeeze(0)
        simp_ind_x1y1 = simplify_coords_idx(x1y1, 1.0)
        simp_ind_x2y2 = simplify_coords_idx(x2y2, 1.0)
        detections_simplified[simp_ind_x1y1, id_ind, slice(0, 3, 2)] = detections[simp_ind_x1y1, id_ind, slice(0, 3, 2)]
        detections_simplified[simp_ind_x2y2, id_ind, slice(1, 4, 2)] = detections[simp_ind_x2y2, id_ind, slice(1, 4, 2)]    

    if not get_video:
        return detections_simplified

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
    print(f'finished {output_video_path}')


if __name__ == '__main__':
    result = pickle.load(open('temp/result', 'rb'))
    output = convert_result(result)
