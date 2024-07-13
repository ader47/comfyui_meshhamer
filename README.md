# ComfyUI-MeshHamer
![](./images/example.png)

## Install
In `ComfyUI` root folder, run the following commands:

```shell
cd custom_nodes
git clone https://github.com/ader47/comfyui_meshhamer.git
cd comfyui_meshhamer
cd mesh_hamer
pip install -e .[all]
cd third-party
git clone https://github.com/ViTAE-Transformer/ViTPose.git
pip install -e .
cd ../..
````

## Download models
```shell
```` 

## Configration
- 目录结构

```shell

```



## TODO 
- Change the `dectron2` model to reduce the VRAM usage.