from qdtrack.apis.inference import init_model, inference_model, export_video
from tools.post_processing import post_processing, convert_result, video_to_images
from glob import glob
import pickle, os, torch
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename


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
    # outputs = streamer.predict(img)
    output = convert_result(result)
    
    #post processing
    view_video = request.form.get('view_video', False)
    result = post_processing(output, video_path, model.CLASSES, get_video=view_video)
    if view_video:
        return send_file(result)
    return jsonify(result)


def test_run(video):
    fname = video.split('/')[-1].replace('.mp4', '_result.mp4')
    if os.path.exists('output/'+fname): return
    result = inference_model(model, video)
    pickle.dump(result, open('temp/result', 'wb'))
    result = pickle.load(open('temp/result', 'rb'))
    output = convert_result(result)
    # output = 'temp/result.csv'
    post_processing(output, video, model.CLASSES, get_video=True)


if __name__ == '__main__':
    model = init_model(
        config='configs/qdtrack-basch.py',
        checkpoint='models/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth',
    )
    # app.run(port=5005, debug=True, host='0.0.0.0')

    # videos = glob('data/bosch_tracking/*.mp4')
    # for video in videos:
    #     test_run(video)

    test_run('data/video_20s.mp4')