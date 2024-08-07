# Consistency Policy
安装依赖
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

首先配置虚拟环境
```console
$ mamba env create -f conda_environment.yaml
```
如果使用wandb监控训练过程，需要login
```console
[consistency-policy]$ conda activate consistency-policy
(consistency-policy)[consistency-policy]$ wandb login
```

## Teacher 模型训练
Techaer 模型基于 EDM 框架搭建，使用如下命令开始训练：
```console
(consistency-policy)[consistency-policy]$ python train.py --config-dir=configs/ --config-name=edm_square.yaml logging.name=edm_square
```
使用的配置文件位于 `configs/edm_square.yaml`

修改数据集路径：修改配置文件中的dataset_path参数，位于146行，181行，187行，这里我修改为 `dataset_path:  ../dataset/image_abs.hdf5`

如果运行提示找不到输出文件路径，手动创建输出文件
```consle
$ touch outputs/edm/square/logs.json.txt
```
如果 wandb 提示连接失败，可以在配置文件中改为离线模式
```yaml
logging:
  group: null
  id: null
  mode: offline # online
  name: dp_test
  project: diffusion_policy_debug
  resume: true
  tags:
  - train_diffusion_unet_hybrid
  - square_image
  - default
```