import requests
import os

ARTICLE_ID_CNN = 28100180
ARTICLE_ID_VIT = 28071419
ARTICLE_ID_UNN = 28100201
BASE_URL = 'https://api.figshare.com/v2'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'weights')

def download_figshare_weights(ARTICLE_ID):
    os.makedirs(DATA_DIR, exist_ok=True)

    response = requests.get(f'{BASE_URL}/articles/{ARTICLE_ID}')
    response.raise_for_status()
    metadata = response.json()

    files = metadata.get('files', [])
    for file_info in files:
        file_name = file_info['name']
        download_url = file_info['download_url']
        file_path = os.path.join(DATA_DIR, file_name)

        print(f'Downloading {file_name}...')
        file_response = requests.get(download_url)
        file_response.raise_for_status()

        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        print(f'Successfully downloaded {file_name}')

if __name__ == '__main__':
    download_figshare_weights(ARTICLE_ID=ARTICLE_ID_CNN)
    download_figshare_weights(ARTICLE_ID=ARTICLE_ID_VIT)
    download_figshare_weights(ARTICLE_ID=ARTICLE_ID_UNN)
