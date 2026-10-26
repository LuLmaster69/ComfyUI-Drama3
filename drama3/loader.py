"""Load the converted Drama 3 files (`tools/convert.py`) as ComfyUI-managed models."""
import contextlib
import json
import logging
import os

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from .codec import Drama3Codec
from .config import parse_config
from .model import Drama3Model, SecondGPU
from .offload import CACHE_RESERVE, ExpertStore, pin_budget

TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish", "tokenizer.json")
# The LM head rows of the newline and <|audio_separator|>, which the converted files leave out: a dialogue's take says
# them between its lines.
LINE_HEAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish", "line_head.safetensors")
# What each card keeps besides its share of the layers: the first card holds the embedding table, the second the
# head, the tower and the caches.
FIRST_RESERVE = 3 * 1024 ** 3
SECOND_RESERVE = 4 * 1024 ** 3
# What a card keeps free besides the layers whose experts it holds when the experts are split between VRAM and RAM: the
# expert cache's reserve for a prompt's activations, then the CUDA context, the kernels' workspace, the decode graphs and
# a small cache. Down to the reserve alone when that lets the host pin the experts left to RAM.
SPLIT_HEADROOM = CACHE_RESERVE + 1024 ** 3
SPLIT_MIN_HEADROOM = CACHE_RESERVE
GIB = 1024 ** 3


class _OpsConfig:
    """`pick_operations` reads `.quant_config` off a model config; quantized files select `mixed_precision_ops`,
    which also carries the `MoEExperts` bank."""

    def __init__(self, quant_config):
        self.quant_config = quant_config


class Drama3:
    """A loaded Drama 3: its patchers, where each part runs, the tokenizer, and the (ids, rows) of the line head."""

    def __init__(self, model, patcher, second=None, second_patcher=None, experts=None):
        from tokenizers import Tokenizer
        self.model = model
        self.patcher = patcher
        self.second = second
        self.second_patcher = second_patcher
        self.experts = experts
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        line_head = comfy.utils.load_torch_file(LINE_HEAD_PATH)
        self.line_head = (line_head["ids"], line_head["weight"])

    @property
    def config(self):
        return self.model.config

    def patchers(self):
        return [p for p in (self.patcher, self.second_patcher) if p is not None]

    def session(self):
        """One generation's hold on the expert cache (experts in RAM): takes inside it share what earlier ones loaded."""
        if self.experts is None:
            return contextlib.nullcontext()
        return self.experts.session(self.patcher.load_device, self.patchers(), cpu_decode=self.second is None)

    def parts(self):
        """(layers, layer devices, final norm, audio head, tower): the head and tower run on the last device."""
        backbone = self.model.model
        first = self.patcher.load_device
        if self.second is None:
            layers = list(backbone.layers)
            return layers, [first] * len(layers), backbone.norm, self.model.audio_head, self.model.tower
        second = self.second_patcher.load_device
        layers = list(backbone.layers) + list(self.second.layers)
        devices = [first] * len(backbone.layers) + [second] * len(self.second.layers)
        return layers, devices, self.second.norm, self.second.audio_head, self.second.tower


def layer_bytes(layer):
    return sum(t.numel() * t.element_size() for t in layer.state_dict().values())


def auto_split(model, first_device, second_device):
    """The first layer index the second card runs: as many layers on the first card as its free memory allows,
    the rest on the second, which also holds the head and the tower."""
    sizes = [layer_bytes(layer) for layer in model.model.layers]
    tail = sum(t.numel() * t.element_size() for m in (model.audio_head, model.tower, model.model.norm) for t in m.state_dict().values())
    budget = comfy.model_management.get_total_memory(first_device) - FIRST_RESERVE - layer_bytes(model.model.embed_tokens)
    second_budget = comfy.model_management.get_total_memory(second_device) - SECOND_RESERVE - tail
    # balance the two cards' headroom
    best = None
    for first in range(1, len(sizes)):
        left = budget - sum(sizes[:first])
        right = second_budget - sum(sizes[first:])
        score = min(left, right)
        if best is None or score > best[0]:
            best = (score, first)
    if best[0] < 0:
        logging.warning("Drama 3: the model does not fit on both cards; ComfyUI will stream the rest of it")
    return best[1]


def expert_sizes(state, layers):
    """(bytes of each layer's routed experts, bytes of everything else) in a model file's state dict."""
    sizes, other = [0] * layers, 0
    for key, tensor in state.items():
        if key.startswith("model.layers.") and ".mlp.experts." in key:
            sizes[int(key.split(".")[2])] += tensor.nbytes
        else:
            other += tensor.nbytes
    return sizes, other


def first_fitting(sizes, budget):
    """How many of the first layers' experts fit in `budget` bytes."""
    count = 0
    while count < len(sizes) and sizes[count] <= budget:
        budget -= sizes[count]
        count += 1
    return count


def vram_expert_layers(sizes, other, device, requested=0):
    """How many of the first layers keep their experts in VRAM when the experts are split between VRAM and RAM.

    `requested`, or with 0 as many as the card holds next to the rest of the model with SPLIT_HEADROOM to spare. When
    the experts that leave in RAM are more than the host can pin (`pin_budget`: about half the RAM on Windows), as many
    more as it takes, down to SPLIT_MIN_HEADROOM to spare: pageable experts turn the decode kernels off."""
    total = comfy.model_management.get_total_memory(device)
    fit = first_fitting(sizes, total - other - SPLIT_HEADROOM)
    most = first_fitting(sizes, total - other - SPLIT_MIN_HEADROOM)
    if requested:
        count = min(requested, len(sizes))
        if count > most:
            logging.warning("Drama 3: the experts of %d layers and the rest of the model take %.1f GiB of the %.1f GiB "
                            "on %s; ComfyUI will stream what does not fit, which is slow. Around %d layers fit.",
                            count, (sum(sizes[:count]) + other) / GIB, total / GIB, device, fit)
        return count
    pinnable = pin_budget()
    needed = 0
    while needed < len(sizes) and sum(sizes[needed:]) > pinnable:
        needed += 1
    if fit < needed <= most:
        logging.info("Drama 3: %d layers keep their experts in VRAM, more than leave %.1f GiB free, so that the host "
                     "can pin the others' (%.1f GiB may be pinned)", needed, SPLIT_HEADROOM / GIB, pinnable / GIB)
        return needed
    return fit


def load_drama3(path, second_device=None, split=0, experts_in_ram=False, expert_cache_gb=0.0, split_experts=False,
                vram_layers=0):
    """`experts_in_ram` keeps every layer's routed experts in host memory behind a VRAM cache; `split_experts` keeps
    the experts of the first `vram_layers` layers (0: as many as fit, `vram_expert_layers`) in VRAM, where ComfyUI
    loads them straight from the file, and only the others' in host memory."""
    if split_experts and second_device is not None:
        raise ValueError("Drama 3: split experts run on one GPU; set second_gpu to none, or keep the experts in VRAM")
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if not metadata or "drama3_config" not in metadata:
        raise ValueError(f"{os.path.basename(path)} is not a converted Drama 3 file (tools/convert.py model)")
    config = parse_config(json.loads(metadata["drama3_config"]))
    sizes, other = expert_sizes(state, config.text.num_hidden_layers)
    config.fp8_experts = metadata.get("drama3_format") == "fp8"
    quant_config = comfy.utils.detect_layer_quantization(state, "")
    dtype = torch.bfloat16
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    operations = comfy.ops.pick_operations(dtype, dtype, load_device=load_device, model_config=_OpsConfig(quant_config))
    model = Drama3Model(config, device=offload_device, dtype=dtype, operations=operations)
    comfy.model_management.archive_model_dtypes(model)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    result = model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
    del state
    if result.unexpected_keys or result.missing_keys:
        raise ValueError(f"Drama 3 checkpoint keys: unexpected {result.unexpected_keys[:6]}, missing {result.missing_keys[:6]}")
    experts = None
    if experts_in_ram or split_experts:
        layers = model.model.layers
        resident = vram_expert_layers(sizes, other, load_device, vram_layers) if split_experts else 0
        if resident < len(layers):
            experts = ExpertStore(int(expert_cache_gb * GIB))
            for index in range(resident, len(layers)):
                layers[index].mlp.experts = experts.add(layers[index].mlp.experts, index)
        if split_experts:
            log_split(sizes, other, resident, load_device)
        if experts is not None and not experts.all_pinned and not comfy.model_management.DISABLE_PINNED_MEMORY:
            logging.warning("Drama 3: %.1f GiB of experts are pinned in RAM and %.1f GiB could not be (Windows pins at "
                            "most about half the RAM, elsewhere what is free): those stay pageable and generation runs "
                            "without the decode kernels, several times slower. Keep more layers' experts in VRAM "
                            "(split VRAM + RAM, vram_expert_layers), free RAM, or take a smaller file.",
                            experts.pinned_bytes / GIB, experts.pageable_bytes / GIB)
    if second_device is None:
        return Drama3(model, patcher, experts=experts)

    first = split or auto_split(model, load_device, second_device)
    second = SecondGPU(model, first)
    second_patcher = comfy.model_patcher.CoreModelPatcher(second, load_device=second_device, offload_device=offload_device)
    logging.info("Drama 3: layers 0-%d on %s, layers %d-%d, head and tower on %s", first - 1, load_device, first,
                 config.text.num_hidden_layers - 1, second_device)
    return Drama3(model, patcher, second, second_patcher, experts)


def log_split(sizes, other, resident, device):
    layers = len(sizes)
    if resident == 0:
        logging.warning("Drama 3: no layer's experts fit in VRAM next to the rest of the model (%.1f GiB); they all stay "
                        "in RAM", other / GIB)
        return
    message = "Drama 3: experts of layers 0-%d in VRAM on %s (%.1f GiB, the rest of the model %.1f GiB)"
    args = [resident - 1, device, sum(sizes[:resident]) / GIB, other / GIB]
    if resident < layers:
        message += ", of layers %d-%d in RAM (%.1f GiB)"
        args += [resident, layers - 1, sum(sizes[resident:]) / GIB]
    logging.info(message, *args)


class Codec:
    def __init__(self, model, patcher):
        self.model = model
        self.patcher = patcher


def load_codec(path):
    state = comfy.utils.load_torch_file(path)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.vae_offload_device()
    model = Drama3Codec(dtype=torch.float32, device=offload_device, operations=comfy.ops.manual_cast)
    comfy.model_management.archive_model_dtypes(model)
    model.load_state_dict(state, strict=True)
    return Codec(model, comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device))
