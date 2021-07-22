import pandas as pd
from glob import glob
import os, random, shutil
from PIL import Image, ImageDraw
from tqdm import tqdm

''' Target format
Generate one txt label file for one image. Each line of the txt label file represents one object. 
The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". 
You can modify src/gen_labels_16.py to generate label files for your custom dataset.

Caltech
   |——————images
   |        └——————00001.jpg
   |        |—————— ...
   |        └——————0000N.jpg
   └——————labels_with_ids
            └——————00001.txt
            |—————— ...
            └——————0000N.txt

In the annotation text, each line is describing a bounding box and has the following format:
[class] [identity] [x_center] [y_center] [width] [height]
The field [class] should be 0. Only single-class multi-object tracking is supported in this version.

The field [identity] is an integer from 0 to num_identities - 1, or -1 if this box has no identity annotation.

*Note that the values of [x_center] [y_center] [width] [height] are normalized by the width/height of the image, so they are floating point numbers ranging from 0 to 1.
'''

''' Source format
    #Values    Name      Description
    ----------------------------------------------------------------------------
    1    frame        Frame within the sequence where the object appearers
    1    track id     Unique tracking id of this object within this sequence
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
    1    truncated    Integer (0,1,2) indicating the level of truncation.
                        Note that this is in contrast to the object detection
                        benchmark where truncation is a float in [0,1].
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.
    
    Sample:
    0 -1 DontCare -1 -1 -10.000000 219.310000 188.490000 245.500000 218.560000 -1000.000000 -1000.000000 -1000.000000 -10.000000 -1.000000 -1.000000 -1.000000
    '''

data_root = 'dataset/kitti_tracking'
ori_img_path = data_root+'/images'
target_img_path = data_root+'/images'
target_txt_path = data_root+'/labels_with_ids'
ori_label_path = data_root+'/original_label/label_02'
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
    'Person': 3
}
ID_class = {v:k for k, v in class_ID.items()}

def get_class_ID(name):
    if name in class_ID:
        return class_ID[name]
    else:
        print(f'ignored class `{name}`')
        return None

def convert_kitti():
    # shutil.rmtree(target_img_path)
    shutil.rmtree(target_txt_path, ignore_errors=True)
    os.makedirs(target_txt_path, exist_ok=True)
    tasks = glob(ori_img_path+'/*')
    tasks = [t.split('/')[-1] for t in tasks]
    print(f'Found images for task: {tasks}')

    last_max_tid = 0
    for task in tasks:
        print(f'Processing task: {task}')
        os.makedirs(f'{target_txt_path}/{task}', exist_ok=True)
        os.makedirs(f'{target_img_path}/{task}', exist_ok=True)
        label_file = glob(f'{ori_label_path}/{task}.txt')[0]
        
        label_df = pd.read_csv(label_file, names=['frame', 'track_id', 'type', 'truncated', 
            'occluded', 'alpha', 'left', 'top', 'right', 'bottom', 'height', 'width', 'length'
                'x', 'y', 'z', 'r_y', 'score'], sep=' ')
        #ge img size
        rand_img = random.choice(glob(f'{ori_img_path}/{task}/*.jpg'))
        img = Image.open(rand_img)
        w, h = img.width, img.height
        label_df['x_center'] = (label_df.left + label_df.right)/2/w
        label_df['y_center'] = (label_df.top + label_df.bottom)/2/h
        label_df['w_p'] = (label_df.right - label_df.left)/w
        label_df['h_p'] = (label_df.bottom - label_df.top)/h
        label_df['class'] = label_df.type.apply(lambda x: get_class_ID(x))
        # drop none class
        # max tid for this seq
        label_df2 = label_df.dropna(how='any').query('track_id != -1')
        max_tid = label_df2.track_id.max()+1
        label_df2.track_id += last_max_tid  # add last_max_tid as base
        print(f'processing seq: {task} with max tid: {max_tid} starting with base: {last_max_tid}')
        last_max_tid += max_tid  # for next seq
        for f in tqdm(range(label_df.frame.max())):
            txt_path = f'{target_txt_path}/{task}/{f:06}.txt'
            img_path = f'{target_img_path}/{task}/{f:06}.jpg'
            if not os.path.exists(img_path):
                img = Image.open(f'{ori_img_path}/{task}/{f:06}.png')
                assert w == img.width and h == img.height
                img.save(img_path)
            # if not os.path.exists(txt_path):
            txt_df = label_df2.query('frame == @f')[['class', 'track_id', 'x_center', 'y_center', 'w_p', 'h_p']]
            txt_df.to_csv(txt_path, header=False, index=False, sep=' ')


# validate generated annotation file
def render_img(img_path, save_path=None):
    img = Image.open(img_path)
    txt = img_path.replace('images', 'labels_with_ids').replace('.jpg', '.txt')
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img, 'RGBA')
    anno = pd.read_csv(txt, names=['class', 'track_id', 'x_center', 'y_center', 'w_p', 'h_p'], sep=' ')
    anno['x1'] = (anno.x_center - anno.w_p/2) * w
    anno['y1'] = (anno.y_center - anno.h_p/2) * h
    anno['x2'] = (anno.x_center + anno.w_p/2) * w
    anno['y2'] = (anno.y_center + anno.h_p/2) * h
    for i, row in anno.iterrows():
        draw.rectangle([(row.x1, row.y1), (row.x2, row.y2)])
        draw.text((row.x1, row.y1), f'{ID_class[row["class"]]}({row.track_id})')
    if not save_path:
        img.show(title='anno')
    else:
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        img.save(save_path)


def show_img(img=None):
    if not img:
        imgs = glob(target_img_path+'/**/*.jpg', recursive=True)
        img = random.choice(imgs)
    frame = img.split('/')[-1].split('.')[0]
    print(f'ploting task: {frame}')
    render_img(img)

def gen_train_val():
    tasks = glob(target_img_path+'/*')
    tasks = [t.split('/')[-1] for t in tasks]
    #split train val
    train_set = tasks[:int(len(tasks)*0.8)]
    val_set = tasks[int(len(tasks)*0.8):]

    with open(f'{data_root}/kitti.train', 'w') as f:
        for task in train_set:
            images = glob(f'{target_img_path}/{task}/*.jpg')
            for t in images:
                f.write(t+'\n')

    with open(f'{data_root}/kitti.val', 'w') as f:
        for task in val_set:
            images = glob(f'{target_img_path}/{task}/*.jpg')
            for t in images:
                f.write(t+'\n')

def render_video():
    rander_path = 'dataset/kitti_tracking/rendered'
    tasks = glob(target_img_path + '/*')
    for task in tasks:
        seq = task.split('/')[-1]
        imgs = glob(f'{target_img_path}/{seq}/*.jpg')
        for img in tqdm(imgs):
            name = img.split('/')[-1]
            render_img(img, f'{rander_path}/{seq}/{name}')
        output_video_path = f'{rander_path}/{seq}.mp4'
        cmd_str = f'ffmpeg -y -framerate 10 -f image2 -i {rander_path}/{seq}/%06d.jpg -c:v libx264 -preset fast -x264-params crf=25 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -r 30 {output_video_path}'
        print(f'>>>Running ffmpeg with cmd: {cmd_str}')
        os.system(cmd_str)
        import shutil
        shutil.rmtree(f'{rander_path}/{seq}/')

if __name__ == '__main__':
    convert_kitti()
    gen_train_val()
    # show_img()
    render_video()
    
