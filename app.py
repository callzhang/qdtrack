from qdtrack.apis.inference import init_model, inference_model, export_video
from tools.post_processing import post_processing, convert_result, video_to_images
from glob import glob
import pickle, os

if __name__ == '__main__':
    videos = glob('data/bosch_tracking/*.mp4')
    for video in videos:
        fname = video.split('/')[-1].replace('.mp4', '_result.mp4')
        if os.path.exists('output/'+fname):
            continue
        # img_folder = video_to_images(video)
        # imgs = sorted(glob(f'{img_folder}/*.jpg'))
        model = init_model(
            config='configs/qdtrack-basch.py',
            checkpoint='models/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth',
        )
        result = inference_model(model, video)
        pickle.dump(result, open('temp/result', 'wb'))
        result = pickle.load(open('temp/result', 'rb'))
        output = convert_result(result)
        # output = 'temp/result.csv'
        post_processing(output, video, model.CLASSES)

        import shutil
        shutil.rmtree(img_folder, ignore_errors=True)
    # export_video(imgs, result, model.CLASSES)
