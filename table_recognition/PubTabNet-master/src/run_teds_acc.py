import os
import sys
import time
import subprocess

if __name__ == "__main__":
    # structure

    for i in range(19, 21):
        subprocess.call("python3 -u ./table_recognition/PubTabNet-master/src/mmocr_teds_acc_mp_PubTabNet_test.py " + str(i), shell=True)

        time.sleep(20)

    # one GPU
    # subprocess.call("CUDA_VISIBLE_DEVICES=3 python3 -u ./table_recognition/table_inference.py 1 0 2", shell=True)

    # time.sleep(60)

    # # recognition
    # subprocess.call("CUDA_VISIBLE_DEVICES=1 python3 -u ./table_recognition/table_inference.py 2 0 1", shell=True)
