pip3 install -r requirements.txt
python3 setup.py develop
mkdir models/
mkdir data/DETRAC/
cp /content/drive/MyDrive/Data/QDTrack/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth models/
cp -R /content/drive/MyDrive/Data/DETRAC/MVI_20011/ data/DETRAC/