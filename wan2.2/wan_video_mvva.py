import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union

from .models.wan_video_dit import WanModel
from .models.model_manager import ModelManager

from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.pipelines.base import BasePipeline
from diffsynth.prompters import WanPrompter
from diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from diffsynth.models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from diffsynth.models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from .models import ModelConfig


class WanVideoReCamMasterPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedModule,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoReCamMasterPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)], dim=1
        )
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    def tensor2video(self, frames):
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        # frames = [Image.fromarray(frame) for frame in frames]
        imgs = []
        for i, frame in enumerate(frames):
            # Ensure PIL-friendly shape: remove singleton channel dims (H,W,1) -> (H,W)
            frame = np.squeeze(frame)

            # Valid shapes now: (H,W) grayscale, (H,W,3) RGB, (H,W,4) RGBA
            if frame.ndim == 2:
                imgs.append(Image.fromarray(frame))
            elif frame.ndim == 3:
                c = frame.shape[2]
                if c == 1:
                    imgs.append(Image.fromarray(frame[:, :, 0]))
                elif c in (3, 4):
                    imgs.append(Image.fromarray(frame))
                else:
                    raise TypeError(f"tensor2video: unsupported channel count {c} for frame {i}, shape after squeeze {frame.shape}")
            else:
                raise TypeError(f"tensor2video: unsupported frame ndim {frame.ndim} for frame {i}, shape after squeeze {frame.shape}")

        return imgs
        # return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(
            input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        # WAN VAE tiled decode does not support batched hidden states for tiled mode.
        # Always decode one sample at a time for robustness (even when batch=1).
        outputs = []
        if torch.is_tensor(latents):
            if latents.dim() == 5:
                bsz = latents.shape[0]
                for b in range(bsz):
                    out = self.vae.decode(latents[b:b+1], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                    if torch.is_tensor(out):
                        outputs.append(out[0])  # (C,T,H,W)
                    elif isinstance(out, (list, tuple)):
                        outputs.append(out[0])
                    else:
                        raise RuntimeError("Unexpected VAE.decode return type")
            elif latents.dim() == 4:
                out = self.vae.decode(latents.unsqueeze(0), device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                outputs.append(out[0] if torch.is_tensor(out) else out[0])
            else:
                raise ValueError("Unsupported latent tensor rank for decode_video")
        elif isinstance(latents, (list, tuple)):
            for item in latents:
                if torch.is_tensor(item) and item.dim() == 5:
                    out = self.vae.decode(item, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                elif torch.is_tensor(item) and item.dim() == 4:
                    out = self.vae.decode(item.unsqueeze(0), device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                else:
                    raise ValueError("Unsupported latent element in list for decode_video")
                outputs.append(out[0] if torch.is_tensor(out) else out[0])
        else:
            raise ValueError("Unsupported latents type for decode_video")

        return outputs if len(outputs) > 1 else outputs[0]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        source_video=None,
        target_camera=None,
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Lazy import to avoid circular
        from .wan_video_mvva import model_fn_wan_video, TeaCache

        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        self.load_models_to_device(["vae"])
        source_video = source_video.to(dtype=self.torch_dtype, device=self.device)
        source_latents = self.encode_video(source_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        batch_size = source_latents.shape[0]

        noise = self.generate_noise(
            source_latents.shape,
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        ).to(dtype=self.torch_dtype, device=self.device)

        if input_video is not None:
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        cam_emb = target_camera.to(dtype=self.torch_dtype, device=self.device)
        if cam_emb.shape[0] != batch_size:
            if cam_emb.shape[0] == 1:
                cam_emb = cam_emb.expand(batch_size, *cam_emb.shape[1:]).contiguous()
            else:
                raise ValueError(f"target_camera batch ({cam_emb.shape[0]}) != source_video batch ({batch_size})")

        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        # Broadcast prompt embeddings over batch if needed
        if "context" in prompt_emb_posi and prompt_emb_posi["context"].shape[0] != batch_size:
            if prompt_emb_posi["context"].shape[0] == 1:
                prompt_emb_posi["context"] = prompt_emb_posi["context"].expand(batch_size, *prompt_emb_posi["context"].shape[1:]).contiguous()
            else:
                raise ValueError("Positive prompt context batch mismatch")
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            if "context" in prompt_emb_nega and prompt_emb_nega["context"].shape[0] != batch_size:
                if prompt_emb_nega["context"].shape[0] == 1:
                    prompt_emb_nega["context"] = prompt_emb_nega["context"].expand(batch_size, *prompt_emb_nega["context"].shape[1:]).contiguous()
                else:
                    raise ValueError("Negative prompt context batch mismatch")

        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
            # Broadcast image embeddings if batch > 1
            if "clip_feature" in image_emb and image_emb["clip_feature"].shape[0] != batch_size:
                if image_emb["clip_feature"].shape[0] == 1:
                    image_emb["clip_feature"] = image_emb["clip_feature"].expand(batch_size, *image_emb["clip_feature"].shape[1:]).contiguous()
                else:
                    raise ValueError("Image clip_feature batch mismatch")
            if "y" in image_emb and image_emb["y"].shape[0] != batch_size:
                if image_emb["y"].shape[0] == 1:
                    image_emb["y"] = image_emb["y"].expand(batch_size, *image_emb["y"].shape[1:]).contiguous()
                else:
                    raise ValueError("Image latent y batch mismatch")
        else:
            image_emb = {}

        extra_input = self.prepare_extra_input(latents)

        tea_cache_posi = {
            "tea_cache": (
                TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
                if tea_cache_l1_thresh is not None
                else None
            )
        }
        tea_cache_nega = {
            "tea_cache": (
                TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
                if tea_cache_l1_thresh is not None
                else None
            )
        }

        self.load_models_to_device(["dit"])
        tgt_latent_length = latents.shape[2]
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.to(dtype=self.torch_dtype, device=self.device)
            if timestep.dim() == 0:
                timestep = timestep.expand(batch_size)
            elif timestep.dim() == 1 and timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)

            latents_input = torch.cat([latents, source_latents], dim=2)
            noise_pred_posi = model_fn_wan_video(
                self.dit,
                latents_input,
                timestep=timestep,
                cam_emb=cam_emb,
                **prompt_emb_posi,
                **image_emb,
                **extra_input,
                **tea_cache_posi,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit,
                    latents_input,
                    timestep=timestep,
                    cam_emb=cam_emb,
                    **prompt_emb_nega,
                    **image_emb,
                    **extra_input,
                    **tea_cache_nega,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(
                noise_pred[:, :, :tgt_latent_length, ...],
                self.scheduler.timesteps[progress_id],
                latents_input[:, :, :tgt_latent_length, ...],
            )

        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])

        # Convert decoded tensors to PIL frames per sample
        videos = []
        if isinstance(frames, (list, tuple)):
            for fr in frames:
                videos.append(self.tensor2video(fr))
        elif torch.is_tensor(frames):
            for b in range(frames.shape[0]):
                videos.append(self.tensor2video(frames[b]))
        else:
            raise RuntimeError("Unexpected decode_video return type for WAN pipeline")

        if batch_size == 1:
            return videos[0]
        return videos

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        pipe = WanVideoReCamMasterPipeline(device=device, torch_dtype=torch_dtype)

        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if use_usp: pipe.enable_usp()
        return pipe


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cam_emb: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    **kwargs,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block in dit.blocks:
            x = block(x, context, cam_emb, t_mod, freqs, (f, h, w))
        if tea_cache is not None:
            tea_cache.store(x)

    t_head = t.unsqueeze(1)
    x = dit.head(x, t_head)
    x = dit.unpatchify(x, (f, h, w))
    return x


