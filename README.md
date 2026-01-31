# AirLLM Setup Guide - Run 70B LLM on 4GB GPU

Run large language models (7B-70B parameters) on a single 4GB GPU using layer-by-layer inference with AirLLM.

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4GB | 4GB+ |
| System RAM | 16GB | 32GB+ |
| Disk Space | 50GB+ | 100GB+ (for model cache) |
| Python | 3.10+ | 3.10.x |
| OS | Windows 10/11, Linux | Windows 11, Ubuntu 22.04 |

## 📦 Installation

### Step 1: Install PyTorch with CUDA

First, check your CUDA version:
```powershell
nvidia-smi
```

Then install PyTorch with the matching CUDA version:

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Verify CUDA Installation

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce GTX 1650  # or your GPU name
```

### Step 3: Install Dependencies

```bash
# Install optimum (MUST use version < 2.0 for BetterTransformer support)
pip install optimum==1.23.1

# Install transformers (compatible version)
pip install transformers==4.45.0

# Install AirLLM
pip install airllm

# Install bitsandbytes for compression (optional but recommended)
pip install bitsandbytes

# Install other dependencies
pip install sentencepiece accelerate safetensors
```

### Step 4: Verify Installation

```bash
pip show airllm optimum transformers torch
```

Expected versions:
```
airllm: 2.x.x
optimum: 1.23.1
transformers: 4.45.0
torch: 2.x.x+cu1xx
```

## 🚀 Quick Start

### Basic Usage

```python
from airllm import AutoModel

MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")

input_text = ['What is the capital of France?']

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False
)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

### With Timing

```python
from airllm import AutoModel
import time

MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")

input_text = ['Hello, how are you?']

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False
)

# Start timing
start_time = time.time()

generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=10,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
end_time = time.time()

print(output)
print(f"Inference time: {end_time - start_time:.2f} seconds")
```

### With 4-bit Compression (3x Faster)

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    compression='4bit'  # Enable 4-bit quantization
)
```

## 📊 Supported Models

| Model | Parameters | VRAM Required | Notes |
|-------|------------|---------------|-------|
| Qwen-7B | 7B | ~4GB | ✅ Tested |
| Llama-2-7B | 7B | ~4GB | ✅ Tested |
| Llama-2-13B | 13B | ~4GB | ✅ Works |
| Llama-2-70B | 70B | ~4GB | ✅ Works (slow) |
| Mistral-7B | 7B | ~4GB | ✅ Tested |
| Llama-3-70B | 70B | ~4GB | ✅ Works |
| Llama-3.1-405B | 405B | ~8GB | ✅ Works (very slow) |

## ⚙️ Configuration Options

```python
model = AutoModel.from_pretrained(
    "model-name",
    compression='4bit',           # '4bit', '8bit', or None
    profiling_mode=True,          # Show timing info
    layer_shards_saving_path="./shards",  # Custom shard path
    hf_token='HF_TOKEN',          # For gated models
    prefetching=True,             # Overlap loading & compute
    delete_original=False         # Delete original after splitting
)
```

## 🔧 Troubleshooting

### Error: `No module named 'optimum.bettertransformer'`

**Cause:** optimum v2.0+ removed BetterTransformer

**Fix:**
```bash
pip uninstall optimum
pip install optimum==1.23.1
```

### Error: `Torch not compiled with CUDA enabled`

**Cause:** CPU-only PyTorch installed

**Fix:** Reinstall PyTorch with CUDA (see Step 1)

### Error: `MetadataIncompleteBuffer`

**Cause:** Out of disk space

**Fix:** Clear HuggingFace cache or extend disk space
```bash
# Clear cache
rm -rf ~/.cache/huggingface/hub/
```

### Error: `401 Client Error - Repo is gated`

**Cause:** Model requires HuggingFace token

**Fix:**
```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    hf_token='YOUR_HF_TOKEN'
)
```

## 📁 File Locations

| Location | Purpose |
|----------|---------|
| `~/.cache/huggingface/hub/` | Downloaded models |
| `models--{org}--{model}/snapshots/.../splitted_model/` | Layer shards |
| `models--{org}--{model}/blobs/` | Original safetensors |

## 🏃 Performance Tips

1. **Use 4-bit compression** for 3x speed improvement
2. **Reduce `max_new_tokens`** for faster responses
3. **Use SSD storage** for model cache (faster layer loading)
4. **Close other GPU applications** to free VRAM

## 📚 How It Works

AirLLM uses **layer-by-layer inference**:

1. Splits model into individual transformer layers
2. Loads one layer at a time into GPU memory
3. Processes the layer and frees memory
4. Repeats for all layers

This allows running models much larger than GPU memory!

```
┌─────────────────────────────────────────────┐
│  Model: 70B parameters (~140GB)             │
│  GPU VRAM: 4GB                              │
│                                             │
│  Layer 1 → Load → Process → Unload          │
│  Layer 2 → Load → Process → Unload          │
│  ...                                        │
│  Layer 80 → Load → Process → Unload         │
│                                             │
│  Result: Full model inference on 4GB GPU!   │
└─────────────────────────────────────────────┘
```

## 📖 References

- [AirLLM GitHub](https://github.com/lyogavin/airllm)
- [AirLLM Paper](https://arxiv.org/abs/2212.09720)
- [HuggingFace Blog Post](https://huggingface.co/blog/lyogavin/airllm)

## 📝 License

AirLLM is released under the Apache 2.0 License.

---

**Created:** January 2026  
**Tested on:** Windows 11, Python 3.10.11, CUDA 12.1, 4GB GPU (GTX 1650)
