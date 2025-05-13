import html
import json
import os
import re
import urllib.parse as ul
from typing import Optional, Tuple, Union

import ftfy
import torch
import torch.distributed as dist
from bs4 import BeautifulSoup
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel

from videosys.core.pab_mgr import PABConfig, set_pab_manager, update_steps
from videosys.core.pipeline import VideoSysPipeline, VideoSysPipelineOutput
from videosys.models.autoencoders.autoencoder_kl_open_sora import OpenSoraVAE_V1_2
from videosys.models.transformers.open_sora_transformer_3d import STDiT3
from videosys.schedulers.scheduling_rflow_open_sora import RFLOW
from videosys.utils.utils import save_video, set_seed

from .data_process import get_image_size, get_num_frames, prepare_multi_resolution_info, read_from_path

os.environ["TOKENIZERS_PARALLELISM"] = "true"


BAD_PUNCT_REGEX = re.compile(
    r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
)  # noqa


class OpenSoraPABConfig(PABConfig):
    def __init__(
        self,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [450, 930],
        spatial_range: int = 2,
        temporal_broadcast: bool = True,
        temporal_threshold: list = [450, 930],
        temporal_range: int = 4,
        cross_broadcast: bool = True,
        cross_threshold: list = [450, 930],
        cross_range: int = 6,
        mlp_broadcast: bool = True,
        mlp_spatial_broadcast_config: dict = {
            676: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            788: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            864: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        },
        mlp_temporal_broadcast_config: dict = {
            676: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            788: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            864: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        },
    ):
        super().__init__(
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
            temporal_broadcast=temporal_broadcast,
            temporal_threshold=temporal_threshold,
            temporal_range=temporal_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_range=cross_range,
            mlp_broadcast=mlp_broadcast,
            mlp_spatial_broadcast_config=mlp_spatial_broadcast_config,
            mlp_temporal_broadcast_config=mlp_temporal_broadcast_config,
        )


class OpenSoraConfig:
    """
    This config is to instantiate a `OpenSoraPipeline` class for video generation.

    To be specific, this config will be passed to engine by `VideoSysEngine(config)`.
    In the engine, it will be used to instantiate the corresponding pipeline class.
    And the engine will call the `generate` function of the pipeline to generate the video.
    If you want to explore the detail of generation, please refer to the pipeline class below.

    Args:
        transformer (str):
            The transformer model to use. Defaults to "hpcai-tech/OpenSora-STDiT-v3".
        vae (str):
            The VAE model to use. Defaults to "hpcai-tech/OpenSora-VAE-v1.2".
        text_encoder (str):
            The text encoder model to use. Defaults to "DeepFloyd/t5-v1_1-xxl".
        num_gpus (int):
            The number of GPUs to use. Defaults to 1.
        num_sampling_steps (int):
            The number of sampling steps. Defaults to 30.
        cfg_scale (float):
            The configuration scale. Defaults to 7.0.
        tiling_size (int):
            The tiling size. Defaults to 4.
        enable_flash_attn (bool):
            Whether to enable Flash Attention. Defaults to False.
        enable_pab (bool):
            Whether to enable Pyramid Attention Broadcast. Defaults to False.
        pab_config (CogVideoXPABConfig):
            The configuration for Pyramid Attention Broadcast. Defaults to `LattePABConfig()`.

    Examples:
        ```python
        from videosys import OpenSoraConfig, VideoSysEngine

        # change num_gpus for multi-gpu inference
        # sampling parameters are defined in the config
        config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
        engine = VideoSysEngine(config)

        prompt = "Sunset over the sea."
        # num frames: 2s, 4s, 8s, 16s
        # resolution: 144p, 240p, 360p, 480p, 720p
        # aspect ratio: 9:16, 16:9, 3:4, 4:3, 1:1
        video = engine.generate(
            prompt=prompt,
            resolution="480p",
            aspect_ratio="9:16",
            num_frames="2s",
        ).video[0]
        engine.save_video(video, f"./outputs/{prompt}.mp4")
        ```
    """

    def __init__(
        self,
        transformer: str = "hpcai-tech/OpenSora-STDiT-v3",
        vae: str = "hpcai-tech/OpenSora-VAE-v1.2",
        text_encoder: str = "DeepFloyd/t5-v1_1-xxl",
        # ======== distributed ========
        num_gpus: int = 1,
        # ======== scheduler ========
        num_sampling_steps: int = 30,
        cfg_scale: float = 7.0,
        # ======= memory =======
        cpu_offload: bool = False,
        # ======== vae ========
        tiling_size: int = 4,
        # ======== speedup ========
        enable_flash_attn: bool = False,
        # ======== pab ========
        enable_pab: bool = False,
        pab_config: PABConfig = OpenSoraPABConfig(),
    ):
        self.pipeline_cls = OpenSoraPipeline
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        # ======== distributed ========
        self.num_gpus = num_gpus
        # ======== scheduler ========
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale
        # ======== vae ========
        self.tiling_size = tiling_size
        # ======= memory ========
        self.cpu_offload = cpu_offload
        # ======== speedup ========
        self.enable_flash_attn = enable_flash_attn
        # ======== pab ========
        self.enable_pab = enable_pab
        self.pab_config = pab_config


class OpenSoraPipeline(VideoSysPipeline):
    r"""
    Pipeline for text-to-image generation using PixArt-Alpha.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`STDiT3`]):
            A text conditioned `STDiT3` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """
    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa
    _optional_components = ["tokenizer", "text_encoder", "vae", "transformer", "scheduler"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        config: OpenSoraConfig,
        text_encoder: Optional[T5EncoderModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        vae: Optional[AutoencoderKL] = None,
        transformer: Optional[STDiT3] = None,
        scheduler: Optional[RFLOW] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self._config = config
        self._device = device
        self._dtype = dtype

        # initialize the model if not provided
        if text_encoder is None:
            text_encoder = T5EncoderModel.from_pretrained(config.text_encoder).to(dtype)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        if vae is None:
            vae = OpenSoraVAE_V1_2(
                from_pretrained=config.vae,
                micro_frame_size=17,
                micro_batch_size=config.tiling_size,
            ).to(dtype)
        if transformer is None:
            transformer = STDiT3.from_pretrained(config.transformer, enable_flash_attn=config.enable_flash_attn).to(
                dtype
            )
        if scheduler is None:
            scheduler = RFLOW(
                use_timestep_transform=True, num_sampling_steps=config.num_sampling_steps, cfg_scale=config.cfg_scale
            )

        # pab
        if config.enable_pab:
            set_pab_manager(config.pab_config)

        self.register_modules(
            text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler, tokenizer=tokenizer
        )

        # cpu offload
        if config.cpu_offload:
            self.enable_model_cpu_offload()
        else:
            self.set_eval_and_device(self._device, vae, transformer, text_encoder)

        # parallel
        self._set_parallel()

    def _set_seed(self, seed):
        if dist.get_world_size() == 1:
            set_seed(seed)
        else:
            set_seed(seed, self.transformer.parallel_manager.dp_rank)

    def _set_parallel(
        self, dp_size: Optional[int] = None, sp_size: Optional[int] = None, enable_cp: Optional[bool] = False
    ):
        # init sequence parallel
        if sp_size is None:
            sp_size = dist.get_world_size()
            dp_size = 1
        else:
            assert (
                dist.get_world_size() % sp_size == 0
            ), f"world_size {dist.get_world_size()} must be divisible by sp_size"
            dp_size = dist.get_world_size() // sp_size

        # transformer parallel
        self.transformer.enable_parallel(dp_size, sp_size, enable_cp)

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        device = self._execution_device
        input_ids = text_tokens_and_mask["input_ids"].to(device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(device)
        with torch.no_grad():
            text_encoder_embs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask

    def encode_prompt(self, text):
        caption_embs, emb_masks = self.get_text_embeddings(text)
        caption_embs = caption_embs[:, None]
        return dict(y=caption_embs, mask=emb_masks)

    def null_embed(self, n):
        null_y = self.transformer.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None].to(self._execution_device)
        return null_y

    @staticmethod
    def _basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(BAD_PUNCT_REGEX, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self._basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def text_preprocessing(self, text, use_text_preprocessing: bool = True):
        if use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = self._clean_caption(text)
            text = self._clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        resolution="480p",
        aspect_ratio="9:16",
        num_frames: int = 51,
        loop: int = 1,
        llm_refine: bool = False,
        negative_prompt: str = "",
        seed: int = -1,
        ms: Optional[str] = "",
        refs: Optional[str] = "",
        aes: float = 6.5,
        flow: Optional[float] = None,
        camera_motion: Optional[float] = None,
        condition_frame_length: int = 5,
        align: int = 5,
        condition_frame_edit: float = 0.0,
        return_dict: bool = True,
        verbose: bool = True,
    ) -> Union[VideoSysPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            resolution (`str`, *optional*, defaults to `"480p"`):
                The resolution of the generated video.
            aspect_ratio (`str`, *optional*, defaults to `"9:16"`):
                The aspect ratio of the generated video.
            num_frames (`int`, *optional*, defaults to 51):
                The number of frames to generate.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            mask_feature (`bool` defaults to `True`): If set to `True`, the text embeddings will be masked.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # == basic ==
        fps = 24
        image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(num_frames)
        self._set_seed(seed)
        update_steps(self._config.num_sampling_steps)

        # == prepare batch prompts ==
        batch_prompts = [prompt]
        ms = [ms]
        refs = [refs]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)

        # == get reference for condition ==
        refs = collect_references_batch(refs, self.vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            "OpenSora", len(batch_prompts), image_size, num_frames, fps, self._device, self._dtype
        )

        # == process prompts step by step ==
        # 0. split prompt
        # each element in the list is [prompt_segment_list, loop_idx_list]
        batched_prompt_segment_list = []
        batched_loop_idx_list = []
        for prompt in batch_prompts:
            prompt_segment_list, loop_idx_list = split_prompt(prompt)
            batched_prompt_segment_list.append(prompt_segment_list)
            batched_loop_idx_list.append(loop_idx_list)

        # 1. refine prompt by openai
        # if llm_refine:
        # only call openai API when
        # 1. seq parallel is not enabled
        # 2. seq parallel is enabled and the process is rank 0
        # if not enable_sequence_parallelism or (enable_sequence_parallelism and coordinator.is_master()):
        #     for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
        #         batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

        # # sync the prompt if using seq parallel
        # if enable_sequence_parallelism:
        #     coordinator.block_all()
        #     prompt_segment_length = [
        #         len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
        #     ]

        #     # flatten the prompt segment list
        #     batched_prompt_segment_list = [
        #         prompt_segment
        #         for prompt_segment_list in batched_prompt_segment_list
        #         for prompt_segment in prompt_segment_list
        #     ]

        #     # create a list of size equal to world size
        #     broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
        #     dist.broadcast_object_list(broadcast_obj_list, 0)

        #     # recover the prompt list
        #     batched_prompt_segment_list = []
        #     segment_start_idx = 0
        #     all_prompts = broadcast_obj_list[0]
        #     for num_segment in prompt_segment_length:
        #         batched_prompt_segment_list.append(
        #             all_prompts[segment_start_idx : segment_start_idx + num_segment]
        #         )
        #         segment_start_idx += num_segment

        # 2. append score
        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = append_score_to_prompts(
                prompt_segment_list,
                aes=aes,
                flow=flow,
                camera_motion=camera_motion,
            )

        # 3. clean prompt with T5
        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = [self.text_preprocessing(prompt) for prompt in prompt_segment_list]

        # 4. merge to obtain the final prompt
        batch_prompts = []
        for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
            batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

        # == Iter over loop generation ==
        video_clips = []
        for loop_i in range(loop):
            # == get prompt for loop i ==
            batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

            # == add condition frames for loop ==
            if loop_i > 0:
                refs, ms = append_generated(
                    self.vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                )

            # == sampling ==
            input_size = (num_frames, *image_size)
            latent_size = self.vae.get_latent_size(input_size)
            z = torch.randn(
                len(batch_prompts), self.vae.out_channels, *latent_size, device=self._device, dtype=self._dtype
            )
            model_args.update(self.encode_prompt(batch_prompts_loop))
            y_null = self.null_embed(len(batch_prompts_loop))

            masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
            samples = self.scheduler.sample(
                self.transformer,
                z=z,
                model_args=model_args,
                y_null=y_null,
                device=self._device,
                progress=verbose,
                mask=masks,
            )
            samples = self.vae(samples.to(self._dtype), decode_only=True, num_frames=num_frames)
            video_clips.append(samples)

        for i in range(1, loop):
            video_clips[i] = video_clips[i][:, dframe_to_frame(condition_frame_length) :]
        video = torch.cat(video_clips, dim=1)

        low, high = -1, 1
        video.clamp_(min=low, max=high)
        video.sub_(low).div_(max(high - low, 1e-5))
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return VideoSysPipelineOutput(video=video)

    def save_video(self, video, output_path):
        save_video(video, output_path, fps=24)


def load_prompts(prompt_path, start_idx=None, end_idx=None):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    prompts = prompts[start_idx:end_idx]
    return prompts


def get_save_path_name(
    save_dir,
    sample_name=None,  # prefix
    sample_idx=None,  # sample index
    prompt=None,  # used prompt
    prompt_as_path=False,  # use prompt as path
    num_sample=1,  # number of samples to generate for one prompt
    k=None,  # kth sample
):
    if sample_name is None:
        sample_name = "" if prompt_as_path else "sample"
    sample_name_suffix = prompt if prompt_as_path else f"_{sample_idx:04d}"
    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix[:50]}")
    if num_sample != 1:
        save_path = f"{save_path}-{k}"
    return save_path


def get_eval_save_path_name(
    save_dir,
    id,  # add id parameter
    sample_name=None,  # prefix
    sample_idx=None,  # sample index
    prompt=None,  # used prompt
    prompt_as_path=False,  # use prompt as path
    num_sample=1,  # number of samples to generate for one prompt
    k=None,  # kth sample
):
    if sample_name is None:
        sample_name = "" if prompt_as_path else "sample"
    save_path = os.path.join(save_dir, f"{id}")
    if num_sample != 1:
        save_path = f"{save_path}-{k}"
    return save_path


def append_score_to_prompts(prompts, aes=None, flow=None, camera_motion=None):
    new_prompts = []
    for prompt in prompts:
        new_prompt = prompt
        if aes is not None and "aesthetic score:" not in prompt:
            new_prompt = f"{new_prompt} aesthetic score: {aes:.1f}."
        if flow is not None and "motion score:" not in prompt:
            new_prompt = f"{new_prompt} motion score: {flow:.1f}."
        if camera_motion is not None and "camera motion:" not in prompt:
            new_prompt = f"{new_prompt} camera motion: {camera_motion}."
        new_prompts.append(new_prompt)
    return new_prompts


def extract_json_from_prompts(prompts, reference, mask_strategy):
    ret_prompts = []
    for i, prompt in enumerate(prompts):
        parts = re.split(r"(?=[{])", prompt)
        assert len(parts) <= 2, f"Invalid prompt: {prompt}"
        ret_prompts.append(parts[0])
        if len(parts) > 1:
            additional_info = json.loads(parts[1])
            for key in additional_info:
                assert key in ["reference_path", "mask_strategy"], f"Invalid key: {key}"
                if key == "reference_path":
                    reference[i] = additional_info[key]
                elif key == "mask_strategy":
                    mask_strategy[i] = additional_info[key]
    return ret_prompts, reference, mask_strategy


def collect_references_batch(reference_paths, vae, image_size):
    refs_x = []  # refs_x: [batch, ref_num, C, T, H, W]
    for reference_path in reference_paths:
        if reference_path == "":
            refs_x.append([])
            continue
        ref_path = reference_path.split(";")
        ref = []
        for r_path in ref_path:
            r = read_from_path(r_path, image_size, transform_name="resize_crop")
            r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
            r_x = r_x.squeeze(0)
            ref.append(r_x)
        refs_x.append(ref)
    return refs_x


def extract_prompts_loop(prompts, num_loop):
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop + 1
                text_list.extend([text] * (end_loop - start_loop))
            prompt = text_list[num_loop]
        ret_prompts.append(prompt)
    return ret_prompts


def split_prompt(prompt_text):
    if prompt_text.startswith("|0|"):
        # this is for prompts which look like
        # |0| a beautiful day |1| a sunny day |2| a rainy day
        # we want to parse it into a list of prompts with the loop index
        prompt_list = prompt_text.split("|")[1:]
        text_list = []
        loop_idx = []
        for i in range(0, len(prompt_list), 2):
            start_loop = int(prompt_list[i])
            text = prompt_list[i + 1].strip()
            text_list.append(text)
            loop_idx.append(start_loop)
        return text_list, loop_idx
    else:
        return [prompt_text], None


def merge_prompt(text_list, loop_idx_list=None):
    if loop_idx_list is None:
        return text_list[0]
    else:
        prompt = ""
        for i, text in enumerate(text_list):
            prompt += f"|{loop_idx_list[i]}|{text}"
        return prompt


MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]


def parse_mask_strategy(mask_strategy):
    mask_batch = []
    if mask_strategy == "" or mask_strategy is None:
        return mask_batch

    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        num_group = len(mask_group)
        assert num_group >= 1 and num_group <= 6, f"Invalid mask strategy: {mask}"
        mask_group.extend(MASK_DEFAULT[num_group:])
        for i in range(5):
            mask_group[i] = int(mask_group[i])
        mask_group[5] = float(mask_group[5])
        mask_batch.append(mask_group)
    return mask_batch


def find_nearest_point(value, point, max_value):
    t = value // point
    if value % point > point / 2 and t < max_value // point - 1:
        t += 1
    return t * point


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i, align=None):
    masks = []
    no_mask = True
    for i, mask_strategy in enumerate(mask_strategys):
        no_mask = False
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        mask_strategy = parse_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            if loop_id != loop_i:
                continue
            ref = refs_x[i][m_id]

            if m_ref_start < 0:
                # ref: [C, T, H, W]
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            if align is not None:
                m_ref_start = find_nearest_point(m_ref_start, align, ref.shape[1])
                m_target_start = find_nearest_point(m_target_start, align, z.shape[2])
            m_length = min(m_length, z.shape[2] - m_target_start, ref.shape[1] - m_ref_start)
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    if no_mask:
        return None
    masks = torch.stack(masks)
    return masks


def append_generated(vae, generated_video, refs_x, mask_strategy, loop_i, condition_frame_length, condition_frame_edit):
    ref_x = vae.encode(generated_video)
    for j, refs in enumerate(refs_x):
        if refs is None:
            refs_x[j] = [ref_x[j]]
        else:
            refs.append(ref_x[j])
        if mask_strategy[j] is None or mask_strategy[j] == "":
            mask_strategy[j] = ""
        else:
            mask_strategy[j] += ";"
        mask_strategy[
            j
        ] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length},{condition_frame_edit}"
    return refs_x, mask_strategy


def dframe_to_frame(num):
    assert num % 5 == 0, f"Invalid num: {num}"
    return num // 5 * 17


OPENAI_CLIENT = None
REFINE_PROMPTS = None
REFINE_PROMPTS_PATH = "assets/texts/t2v_pllava.txt"
REFINE_PROMPTS_TEMPLATE = """
You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task. Here are some examples of refined prompts:
{}

The refined prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The refined prompt should be in English.
"""
RANDOM_PROMPTS = None
RANDOM_PROMPTS_TEMPLATE = """
You need to generate one input prompt for video generation task. The prompt should be suitable for the task. Here are some examples of refined prompts:
{}

The prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The prompt should be in English.
"""


def get_openai_response(sys_prompt, usr_prompt, model="gpt-4o"):
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        from openai import OpenAI

        OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    completion = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },  # <-- This is the system message that provides context to the model
            {
                "role": "user",
                "content": usr_prompt,
            },  # <-- This is the user message for which the model will generate a response
        ],
    )

    return completion.choices[0].message.content


def get_random_prompt_by_openai():
    global RANDOM_PROMPTS
    if RANDOM_PROMPTS is None:
        examples = load_prompts(REFINE_PROMPTS_PATH)
        RANDOM_PROMPTS = RANDOM_PROMPTS_TEMPLATE.format("\n".join(examples))

    response = get_openai_response(RANDOM_PROMPTS, "Generate one example.")
    return response


def refine_prompt_by_openai(prompt):
    global REFINE_PROMPTS
    if REFINE_PROMPTS is None:
        examples = load_prompts(REFINE_PROMPTS_PATH)
        REFINE_PROMPTS = REFINE_PROMPTS_TEMPLATE.format("\n".join(examples))

    response = get_openai_response(REFINE_PROMPTS, prompt)
    return response


def has_openai_key():
    return "OPENAI_API_KEY" in os.environ


def refine_prompts_by_openai(prompts):
    new_prompts = []
    for prompt in prompts:
        try:
            if prompt.strip() == "":
                new_prompt = get_random_prompt_by_openai()
                print(f"[Info] Empty prompt detected, generate random prompt: {new_prompt}")
            else:
                new_prompt = refine_prompt_by_openai(prompt)
                print(f"[Info] Refine prompt: {prompt} -> {new_prompt}")
            new_prompts.append(new_prompt)
        except Exception as e:
            print(f"[Warning] Failed to refine prompt: {prompt} due to {e}")
            new_prompts.append(prompt)
    return new_prompts


def add_watermark(
    input_video_path, watermark_image_path="./assets/images/watermark/watermark.png", output_video_path=None
):
    # execute this command in terminal with subprocess
    # return if the process is successful
    if output_video_path is None:
        output_video_path = input_video_path.replace(".mp4", "_watermark.mp4")
    cmd = f'ffmpeg -y -i {input_video_path} -i {watermark_image_path} -filter_complex "[1][0]scale2ref=oh*mdar:ih*0.1[logo][video];[video][logo]overlay" {output_video_path}'
    exit_code = os.system(cmd)
    is_success = exit_code == 0
    return is_success
