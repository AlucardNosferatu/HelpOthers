import os
import pickle
import random

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
# noinspection PyUnresolvedReferences
from urllib3.packages.six import BytesIO

from Config import num_steps, rgb_channel
from DL.cfg import img_shape, batch_size
from Utilities import timestep_tensor, add_noise


def load_prompt_from_csv(prompt_csv_filepath, skip_download=0, download_image=True, load_length=1000):
    print('Current file:', prompt_csv_filepath)
    data_csv = pd.read_csv(prompt_csv_filepath)
    inputs, outputs = [], []
    id_list = []
    for index, row in data_csv.iterrows():
        img_id = row['id']
        img_id_str = str(img_id)
        has_jpg = os.path.exists(os.path.join('Data/Image', img_id_str + '.JPEG'))
        has_png = os.path.exists(os.path.join('Data/Image', img_id_str + '.PNG'))
        if not has_png and not has_jpg:
            if download_image:
                if img_id > skip_download:
                    img_url = 'http:' + row['sample_url']
                    response = requests.get(img_url)
                    if response.status_code != 200:
                        continue
                    content = response.content
                    bytes_io_obj = BytesIO()
                    bytes_io_obj.write(content)
                    img_pil = Image.open(bytes_io_obj)
                    if img_pil.format not in ['JPEG', 'PNG']:
                        continue
                    img_pil.save(open(os.path.join('Data/Image', img_id_str + '.' + img_pil.format), 'wb'))
                else:
                    continue
            else:
                continue
        tags = row['tags']
        inputs.append(tags.split(' '))
        outputs.append(tags.split(' '))
        id_list.append(img_id_str)
        if len(id_list) % 10 == 0:
            print(len(id_list))
        if len(id_list) >= load_length:
            break
    return inputs, outputs, id_list


def save_prompt_to_txt(prompt_txt_filepath, inputs, outputs, id_list):
    lines = []
    for id_, input_, output_ in zip(id_list, inputs, outputs):
        line = '\t\t\t'.join([id_, ' '.join(input_), ' '.join(output_)]) + '\n'
        lines.append(line)
    with open(prompt_txt_filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def load_prompt_from_txt(prompt_txt_filepath):
    with open(prompt_txt_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    id_list, inputs, outputs = [], [], []
    for line in lines:
        line = line.strip('\n').split('\t\t\t')
        id_list.append(line[0])
        inputs.append(line[1].split(' '))
        outputs.append(line[2].split(' '))
    return inputs, outputs, id_list


def load_image_from_files(img_dir='Data/Image', empty_ctx_path='Save/empty_context.npy'):
    context = np.squeeze(np.load(empty_ctx_path))
    img_files = os.listdir(img_dir)
    ids = [file.split('.')[0] for file in img_files]
    img_files = [os.path.join(img_dir, file) for file in img_files]
    timesteps = np.arange(1, 1000, 1000 // num_steps)
    x_wn = []
    x_tt = []
    x_ct = []
    y = []
    for i in tqdm(range(len(img_files))):
        img_file = img_files[i]
        image = cv2.resize(cv2.imread(img_file), img_shape)
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image = np.array(cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB))
        image = image.astype('float32') / 255.0

        timestep = random.choice(timesteps)
        ts_tensor = np.squeeze(timestep_tensor(1, timestep))

        with_noise = np.squeeze(
            add_noise(np.reshape(image, (1, img_shape[0], img_shape[1], rgb_channel)), timestep))

        x_wn.append(with_noise)
        x_tt.append(ts_tensor)
        x_ct.append(context)
        y.append(image)

        if len(y) >= batch_size:
            input_pkl = [np.array(x_wn), np.array(x_tt), np.array(x_ct)]
            pickle.dump(input_pkl, open(os.path.join('Data/Array/Input', ids[i] + '.pkl'), 'wb'))
            np.save(os.path.join('Data/Array/Output', ids[i] + '.npy'), np.array(y))
            x_wn.clear()
            x_tt.clear()
            x_ct.clear()
            y.clear()


def read_ids():
    ids = os.listdir('Data/Array/Input')
    ids = [id_.split('.')[0] for id_ in ids]
    return ids


def generator_train():
    ids_ = read_ids()
    while True:
        id_ = random.choice(ids_)
        filepath = os.path.join('Data/Array/Input', id_ + '.pkl')
        input_list = pickle.load(open(filepath, 'rb'))
        filepath = os.path.join('Data/Array/Output', id_ + '.npy')
        output_array = np.load(filepath)
        yield input_list, output_array


if __name__ == '__main__':
    # i, o, ids = load_prompt_from_csv('Data/all_data.csv', skip_download=1341, download_image=True, load_length=1000)
    # save_prompt_to_txt('Data/Prompt.txt', i, o, ids)
    # i, o, ids = load_prompt_from_txt('Data/Prompt.txt')
    # tok, v_size = task_conv_chn(None, None, False, False)
    # load_image_from_files()
    g_in = generator_train()
    g_in.__next__()
