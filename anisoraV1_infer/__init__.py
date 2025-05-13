import logging,time,re
from fastercache_sample_cogvideox_sp import Args
from videosys import VideoSysEngine
import pdb

def append_score_to_prompts(prompt, aes=None, flow=None, camera_motion=None):
    new_prompt = prompt.split('camera motion:')[0]
    if aes is not None and "aesthetic score:" not in prompt:
        new_prompt = f"{new_prompt} aesthetic score: {aes:.1f}."
    if flow is not None:
        if "motion score:" not in prompt:
            #add
            new_prompt = f"{new_prompt} motion score: {flow:.1f}."
        if "motion score:" in prompt:
            #rewrite
            new_prompt = re.sub(r"motion score: \d+(\.\d+)?", f"motion score: {flow:.1f}", new_prompt)
    if "camera motion:" in prompt:
        cm = prompt.split('camera motion:')[1]
        new_prompt = f"{new_prompt} camera motion: {cm}"
    return new_prompt


class CVModel:
    # 实例初始化，模型加载等一次性工作在这里
    # 模型文件地址的根目录为 /data, 具体路径可以添加备注说明，例如 /data/app/rubick-audio-analyser/xxx.pth
    def __init__(self,n_gpus):
        # ======================================================
        # 1. cfg and init distributed env
        # ======================================================
        # 超级加倍
        self.engine = VideoSysEngine(Args(num_gpus=n_gpus))

    # run 方法, 模型入口
    # resource 为本地地址，支持传入关键字参数
    def run(self, resource,**kwargs):
        result = {}

        image_path = resource
        # user define
        prompt = kwargs.get('prompt')
        output_path = kwargs.get('output_path')
        gen_len = kwargs.get('gen_len')
        Motion_L = kwargs.get('Motion')
        seed = kwargs.get('seed')
        #根据生成秒数，调整num_frames
        if gen_len=='3':
            num_frames = 49
        elif gen_len=="6":
            num_frames = 97

        start = time.time()

        # 根据flag修改motion score
        if isinstance(Motion_L, float) and Motion_L is not None:
            prompt =  append_score_to_prompts(prompt, aes=5.5, flow=Motion_L, camera_motion=None)
        else:
            # 没有指令且没有值
            if "motion score:" not in prompt:
                prompt = append_score_to_prompts(prompt, aes=5.5, flow=0.7, camera_motion=None)
            else:
                prompt = append_score_to_prompts(prompt, aes=5.5, flow=None, camera_motion=None)

        prompt = [prompt]

        print(prompt)
        logging.info(prompt)
        logging.info(image_path)

        # ======================================================
        # 4. inference
        # ======================================================
        # resource[0]
        self.engine.generate(extra_args={"prompt":prompt,"image_path":image_path,"output_path":output_path,"num_frames":num_frames,"seed":seed})
        end = time.time()
        logging.info(f"[Profiling][Image2Video] {end - start:.6f}s")
        print(f"[Profiling][Image2Video] {end - start:.6f}s")

        result['status'] = 0
        # result['video_url'] = output_list

        return result  # pack_output_url

    def post_run(self):
        # 如有必要对结果作后处理, 模型同学一般不关注
        pass
