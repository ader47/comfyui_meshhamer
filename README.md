# ComfyUI-MeshHamer
![](./images/example.png)

## Installation
Before installation, you should install the [`CUDA_Toolkit`](https://developer.nvidia.com/cuda-toolkit-archive) first.

Enter to the `ComfyUI` root folder, run the following commands:

```shell
cd custom_nodes
git clone --recursive https://github.com/ader47/comfyui_meshhamer.git
cd comfyui_meshhamer
cd mesh_hamer
pip install -e .[all]
cd third-party
pip install -e .
````

## Configration

### Download models
You need to download the pretrained models:

```bash ./mesh_hamer/fetch_demo_data.sh```

Besides these files, you also need to download the MANO model. 
Please visit the MANO website and register to get access to the downloads section. 
We only require the right hand model. 
You need to put `MANO_RIGHT.pkl` under the `mesh_hamer/_DATA/data/mano` folder.

The checkpoints and model config files should be placed in the following structure:
```shell
- comfyui_meshhamer
    - mesh_hamer
      - __DATA
        - data
          - mano
              MANO_RIGHT.pkl
          - mano_mean_params.npz
        - hamer_ckpts
          - checkpoints
              hamer.ckpt
          dataset_config.yaml
          model_config.yaml
        - vitpose_ckpts
          - vitpose+_huge
            wholebody.pth
```
### Config file
In `config.py` file, you should change `CACHE_DIR_HAMER`, `DETECTRON2_INIT_CHECKPOINT` and `MESH_HAMER_CHECKPOINT` to your own path.
The `DEVICE` can be set to `cpu` or `cuda` to use CPU or GPU respectively.

### **Notice**
**This pipline needs about 10GB VRAM**. If you have a GPU with less than 10GB VRAM, you can try to change `DEVICE` in the `config.py` file.
## TODO 
- Change the `dectectron2` or `ViTPose` to reduce the VRAM usage.
- Add external detector api.

# Acknowledgements
Parts of the code are borrowed from the following repositories:
- **[Hamer](https://github.com/geopavlakos/hamer/tree/main)**
- **[controlnet_aux](https://github.com/huggingface/controlnet_aux)**