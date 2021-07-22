from qdtrack.apis.inference import init_model, inference_model, export_video
from tools.post_processing import post_processing, convert_result, merge_result
from glob import glob
import pickle, os, torch, sys, logging
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
# fairmot
from src import _init_paths
import datasets.dataset.jde as datasets
from opts import opts
from track import eval_seq
from tracking_utils.log import logger

logger.setLevel(logging.INFO)
class_ID = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Bus': 4,
    'Misc': 5,
    'DontCare': 6,
    'ignored': 7
}
ID_class = {v: k for k, v in class_ID.items()}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 50  # 50MB max
# app.config['UPLOAD_EXTENSIONS'] = ['.mp4']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None

@app.route("/tracking", methods=["POST"])
def track():
    file = request.files['video']
    assert file.content_type[:6] == 'video/', 'Invalid content_type, must upload a video file (*.mp4)'
    filename = secure_filename(file.filename)
    #TODO: pass link from aliyun OSS
    # save to temp folder
    video_path = f'temp/{filename}'
    os.makedirs('temp', exist_ok=True)
    file.save(video_path)

    # inference
    result = inference_model(model, video_path)
    output = convert_result(result)
    
    #post processing
    view_video = request.form.get('view_video', False)
    result = post_processing(output, video_path, model.CLASSES, get_video=view_video)
    if view_video:
        return send_file(result)
    return jsonify(result)


@app.route("/dual_tracking", methods=["POST"])
def dual_track():
    '''
    track a video with qdtrack and fairmot
    '''
    file = request.files['video']
    assert file.content_type[:6] == 'video/', 'Invalid content_type, must upload a video file (*.mp4)'
    filename = secure_filename(file.filename)
    #TODO: pass link from aliyun OSS
    # save to temp folder
    video_path = f'temp/{filename}'
    os.makedirs('temp', exist_ok=True)
    file.save(video_path)

    # inference vihicle
    result_dict_1 = inference_model(model, video_path)
    result1 = convert_result(result_dict_1)
    # inference perdestrain from fairmot
    result2 = track_fairmot(video_path)
    # merge result
    output = video_path.replace('.mp4', '.csv')
    merge_result(result1, result2, output)

    #post processing
    view_video = request.form.get('view_video', False)
    result = post_processing(output, video_path, ID_class, get_video=view_video)
    if view_video:
        return send_file(result)
    return jsonify(result)


def test_run(video):
    fname = video.split('/')[-1].replace('.mp4', '_result.mp4')
    # if os.path.exists('output/'+fname): return
    result = inference_model(model, video)
    # pickle.dump(result, open('temp/result', 'wb'))
    # result = pickle.load(open('temp/result', 'rb'))
    output = convert_result(result)
    # output = 'temp/result.csv'
    post_processing(output, video, model.CLASSES, get_video=True)


def track_fairmot(video):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init(
        [
            '--load_model', 'models/fairmot_dla34.pth',
            '--input_video', 'videos/video.mp4',
            '--output_root', 'output',
            '--num_classes', '1',
            '--dataset', 'kitti'
        ])
    result_filename = opt.result_file
    print('Starting tracking...')
    dataloader = datasets.LoadVideo(video, opt.img_size)
    fps = dataloader.frame_rate
    opt.track_buffer = fps * opt.track_duration
    opt.fps = fps
    eval_seq(opt, dataloader, opt.dataset, result_filename,
             save_dir=None, show_image=opt.show_image, frame_rate=fps,
             use_cuda=opt.gpus != [-1])
    return result_filename


if __name__ == '__main__':
    model = init_model(
        config='configs/qdtrack-basch.py',
        checkpoint='models/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth',
        # config='configs/tao/qdtrack_frcnn_r50_fpn_24e_lvis.py',
        # checkpoint='models/qdtrack_tao.pth',
    )
    app.run(port=5005, debug=True, host='0.0.0.0')

    # videos = glob('data/bosch_tracking/*.mp4')
    # for video in videos:
    #     test_run(video)

    # test_run('data/video_10s.mp4')
    # test_run('data/video.mp4')
    # track_fairmot('data/video_10s.mp4')
