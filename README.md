

## Debian环境部署

设置代理环境
```bash
> export http_proxy=192.168.2.199:58591
> export https_proxy=192.168.2.199:58591
```

```bash
>  pip install --upgrade diffusers
> sudo apt-get -y install cuda-toolkit-12-3
```

设置python虚拟环境
```bash
> sudo apt install python3-venv  python3-pip
> mkdir /opt/Data/PythonVenv
> cd /opt/Data/PythonVenv
> python3 -m venv Diffusers
> source /opt/Data/PythonVenv/Diffusers/bin/activate
```

部署推理环境
```bash
> pip install --upgrade diffusers -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install accelerate transformers==4.33.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("/opt/Data/THUDM/MagicAnimate/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```



启动丘比特
```bash
> jupyter notebook
> jupyter notebook --no-browser --port 7000 --ip=192.168.2.200
```
