import time
import subprocess

if __name__ == "__main__":

    # structure

    # subprocess.call("CUDA_VISIBLE_DEVICES=0 python3 -u ./table_recognition/table_inference_PubTabNet.py 2 0 2 17 &"
    #                 "CUDA_VISIBLE_DEVICES=1 python3 -u ./table_recognition/table_inference_PubTabNet.py 2 1 2 17", shell=True)
    #
    # time.sleep(20)

    subprocess.call("CUDA_VISIBLE_DEVICES=2 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 0 20 test &"
                    "CUDA_VISIBLE_DEVICES=2 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 1 20 test &"
                    "CUDA_VISIBLE_DEVICES=3 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 2 20 test &"
                    "CUDA_VISIBLE_DEVICES=3 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 3 20 test &"
                    "CUDA_VISIBLE_DEVICES=0 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 4 20 test &"
                    "CUDA_VISIBLE_DEVICES=0 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 5 20 test &"
                    "CUDA_VISIBLE_DEVICES=1 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 6 20 test &"
                    "CUDA_VISIBLE_DEVICES=1 python3 -u ./table_recognition/table_inference_FinTabNet_VQAonBD2023.py 8 7 20 test", shell=True)

    time.sleep(60)

    # subprocess.call("CUDA_VISIBLE_DEVICES=3 python3 -u ./table_recognition/table_inference_WikiTabNet.py 1 0 2 20 test", shell=True)
    #
    # time.sleep(20)
