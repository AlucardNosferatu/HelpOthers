import os

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
# noinspection PyUnresolvedReferences
from urllib3.packages.six import BytesIO


def load_prompt(prompt_csv_filepath):
    print('Current file:', prompt_csv_filepath)
    data_csv = pd.read_csv(prompt_csv_filepath)
    inputs, outputs = [], []
    for index, row in tqdm(data_csv.iterrows()):
        img_id = str(row['id'])
        has_jpg = os.path.exists(os.path.join('Data/Image', img_id + '.JPEG'))
        has_png = os.path.exists(os.path.join('Data/Image', img_id + '.PNG'))
        if not has_png and not has_jpg:
            img_url = 'http:' + row['sample_url']
            response = requests.get(img_url)
            if response.status_code != 200:
                continue
            content = response.content
            bytes_io_obj = BytesIO()
            bytes_io_obj.write(content)
            img_pil = Image.open(bytes_io_obj)
            assert img_pil.format in ['JPEG', 'PNG']
            img_pil.save(open(os.path.join('Data/Image', img_id + '.' + img_pil.format), 'wb'))
        tags = row['tags']
        inputs.append(tags.split(' '))
        outputs.append(tags.split(' '))
    return inputs, outputs


if __name__ == '__main__':
    load_prompt('Data/all_data.csv')
