import time
import sys

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


import urllib.request
urllib.request.urlretrieve("https://storage.googleapis.com/kaggle-competitions-data/kaggle/7880/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521965737&Signature=gTrc8Nb8jg%2Be%2FHomWPa7zJ%2F%2FXHb0RQ4bMHqDVtxipFbJNs2Z8%2FtUclSiAO79fDgq3%2BJJ5rx1a6inkUt0PAN83v1HfRp4YHoIAuF%2BVS7aZz5IMkgutW7hiEVVMxwLhVzu6WihHKIaaPH8CCPu416x3R%2F8wSzj8djRdXDpctTeifn%2FDf%2BrPoUA1b03ITE9P00x0hJ489vF1yUXQ7%2B9btOrOpvn4LXvTihrFQ2Y5%2F%2BkDEH8cS2aOzRrVzEmyHWrJk5c%2F1OiRQCgHTz2rFDdt%2B2yo%2BBsNozn9gu5oJHJ2L9z0ukGzFeggCToyD2eb0S%2FkW5gFEAgF7D7DNHPUyawabqdrg%3D%3D", "train.zip", reporthook)
urllib.request.urlretrieve("https://storage.googleapis.com/kaggle-competitions-data/kaggle/7880/test.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1521965744&Signature=Q%2FAltjVfE8Tvht%2FLZMcxOyNhI0V9EIKRq%2Bc898mS9Hp7hzUa7F22PgZkQUJX3RtNV8bstHrg4%2FHthsIOIBRYZOpGISZzP9oWjYrA5gWSsdFOq7qU8LLr38%2FYKDDkg6jjwJVRtfc5nkPGmPPH0eHlpdK3mJgbGXW9qOV1z%2FPMjwVzBj%2FOhCoN9nWJ0TPMh7YygafAGvj0m0Lgsalp0dq0bVVrgbEkcV1DRtZBuYt%2F6hH3MEZB8zsBz55Oay%2F292vfy%2BIV1ipXXJJZkI3xlNXPNk0Tleg1f7tWxrbiKE5KmEKLnmrzmoPOXrheH2TytCkduUHswuCI%2F8eaxJjWB%2FYorw%3D%3D", "test.zip", reporthook)