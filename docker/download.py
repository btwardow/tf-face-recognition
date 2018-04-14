import requests


if __name__ == '__main__':

    print("Downloading pretrained model for MTCNN...")

    for i in range(1, 4):
        f_name = 'det{}.npy'.format(i)
        print("Downloading: ", f_name)
        url = "https://github.com/davidsandberg/facenet/raw/" \
              "e9d4e8eca95829e5607236fa30a0556b40813f62/src/align/det{}.npy".format(i)
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768

        with open(f_name, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
