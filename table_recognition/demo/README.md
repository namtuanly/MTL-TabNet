  <h1 align="left">MTL-TabNet Demo</h1>

This demo script will give a end to end result from a table image input. The checkpoints accuracy we use in this demo, were described in the **pretrained  model** of this repo's README.



### How to run

1. Modify the config and checkpoint path in **ArgumentParser**.

   You can find config in folder **/MTL-TabNet/configs/** :

   **Table Recognition (MTL-TabNet)** config : table_master_ResnetExtract_Ranger_0705_cell150_batch4.py

   You can also download checkpoints in **Pretrain Model** of <a href="../../README.md">README.md</a>

2. Run script. And you will get the table recognition result in a **HTML** file, which saved in the output dir.

```shell
cd /MTL-TabNet
python ./table_recognition/demo/demo.py
```
