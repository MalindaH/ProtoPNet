import pandas as pd
from PIL import Image
from pathlib import Path

folder_path = "/scratch/gpfs/lh9998/ProtoPNet/CUB_200_2011/"
images_path = folder_path + "images/"
imagestxt_path = folder_path + "images.txt"
tts_path = folder_path + "train_test_split.txt"
bbx_path = folder_path + "bounding_boxes.txt"
# save_path_train = "../datasets/cub200_cropped/train_cropped/"
# save_path_test = "../datasets/cub200_cropped/test_cropped/"
save_path_train = "../datasets/cub200_full/train/"
save_path_test = "../datasets/cub200_full/test/"

images_df = pd.read_csv(imagestxt_path, sep=' ', header=None, names=['id', 'img_file'], index_col=0)
tts_df = pd.read_csv(tts_path, sep=' ', header=None, names=['id', 'is_train'], index_col=0)
bbx_df = pd.read_csv(bbx_path, sep=' ', header=None, names=['id', 'x', 'y', 'width', 'height'], index_col=0)
df = images_df.join(tts_df, on='id').join(bbx_df, on='id')
print(df)

for row in df.iterrows():
    img_file = row[1]['img_file']
    # print(img_file)
    is_train = row[1]['is_train']
    x, y, w, h = row[1]['x'], row[1]['y'], row[1]['width'], row[1]['height']
    # print(is_train, x,y,w,h)
    im = Image.open(images_path + img_file)
    # im1 = im.crop((x, y, x+w, y+h))
    save_path = save_path_test + img_file
    if is_train:
        save_path = save_path_train + img_file
    Path(save_path[:save_path.rfind('/')]).mkdir(parents=True, exist_ok=True)
    # im1.save(save_path)
    im.save(save_path)
    