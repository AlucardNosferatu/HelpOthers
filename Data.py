import os

import pandas as pd
import requests
from PIL import Image
# noinspection PyUnresolvedReferences
from urllib3.packages.six import BytesIO


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
        if len(id_list) > load_length:
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
    pass


if __name__ == '__main__':
    i, o, ids = load_prompt_from_csv('Data/all_data.csv', skip_download=1145, download_image=True)
    save_prompt_to_txt('Data/Prompt.txt', i, o, ids)
