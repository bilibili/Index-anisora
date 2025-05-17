pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
cd character/samurai/sam2  
pip install -e ".[notebooks]" 
cd -
pip install transformers 
pip install opencv-python-headless
pip install loguru
pip install scipy
pip install decord
pip install requests
pip install pyyaml
pip install ruamel.yaml
pip install fairscale
pip install pycocoevalcap
pip install timm
pip install av
pip install git+https://github.com/openai/CLIP.git
pip install importlib_metadata
pip install mmcv==2.1.0
pip install mmaction2 
pip install mmengine
cp src/drn/ /root/miniconda3/envs/AnimeReward/lib/python3.10/site-packages/mmaction/models/localizers/ -r
