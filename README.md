# Robust-VRP
Due to upload restrictions, more complete projects including saved models can be referenced: https://drive.google.com/drive/folders/1nnZvxmeSdWrgihmc9epwNJ8Vch8SxS1S?usp=drive_link

Single GPU can directly run `python RTSP_train.py` or `python RTSP_test.py`

Multiple GPUs need 
```
RTSP_train.py:
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'   # numbers of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # id of GPUs
```
```
cd ./Robust-VRP/Robust-TSP/TSP/RTSP
python -m torch.distributed.launch --nproc_per_node=3 --master_port=11267 RTSP_train.py
```

Single GPU and multi-GPU are adjusted according to the scale and memory size, and the corresponding `RTSPTrainer.py`, `RTSP_train.py`, `utils.py` need to be modified.

# 补充

数据和训练结果中，文件命名中的20、50是problem size，就是图中的节点数量。100、1000是每个点坐标取值范围的上限，但CVRP的问题定义文件里还没实现这个，固定取值上限是10。