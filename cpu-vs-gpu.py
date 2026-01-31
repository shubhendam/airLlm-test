import time
import torch
import gc

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "Qwen/Qwen-7B"
MAX_LENGTH = 128
MAX_NEW_TOKENS = 10
INPUT_TEXT = ['Hello, how are you?']

# ============================================================
# BENCHMARK 1: AirLLM (Layer-by-Layer GPU Inference)
# ============================================================
print("=" * 60)
print("BENCHMARK 1: AirLLM (Layer-by-Layer on 4GB GPU)")
print("=" * 60)

from airllm import AutoModel as AirLLMAutoModel

# Load model with AirLLM
airllm_load_start = time.time()
airllm_model = AirLLMAutoModel.from_pretrained(MODEL_NAME)
airllm_load_end = time.time()

# Tokenize
input_tokens_airllm = airllm_model.tokenizer(
    INPUT_TEXT,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False
)

# Inference
airllm_inference_start = time.time()
generation_output_airllm = airllm_model.generate(
    input_tokens_airllm['input_ids'].cuda(), 
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    return_dict_in_generate=True
)
output_airllm = airllm_model.tokenizer.decode(generation_output_airllm.sequences[0])
airllm_inference_end = time.time()

print(f"\nOutput: {output_airllm}")
print(f"Model Load Time: {airllm_load_end - airllm_load_start:.2f} seconds")
print(f"Inference Time: {airllm_inference_end - airllm_inference_start:.2f} seconds")
print(f"Total Time: {airllm_inference_end - airllm_load_start:.2f} seconds")

# Store results
airllm_results = {
    'load_time': airllm_load_end - airllm_load_start,
    'inference_time': airllm_inference_end - airllm_inference_start,
    'total_time': airllm_inference_end - airllm_load_start,
    'output': output_airllm
}

# Clean up GPU memory
del airllm_model
del generation_output_airllm
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 60)
print("BENCHMARK 2: Transformers (Full Model on CPU - 64GB RAM)")
print("=" * 60)

# ============================================================
# BENCHMARK 2: Standard Transformers (Full Model on CPU)
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model on CPU
cpu_load_start = time.time()
tokenizer_cpu = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model_cpu = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="cpu",  # Force CPU only
    trust_remote_code=True,
    torch_dtype=torch.float32  # Use float32 for CPU
).eval()
cpu_load_end = time.time()

# Tokenize
inputs_cpu = tokenizer_cpu(
    INPUT_TEXT[0], 
    return_tensors='pt',
    truncation=True,
    max_length=MAX_LENGTH
)
# Keep on CPU (no .cuda())

# Inference
cpu_inference_start = time.time()
with torch.no_grad():
    generation_output_cpu = model_cpu.generate(
        **inputs_cpu,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True
    )
output_cpu = tokenizer_cpu.decode(generation_output_cpu[0], skip_special_tokens=True)
cpu_inference_end = time.time()

print(f"\nOutput: {output_cpu}")
print(f"Model Load Time: {cpu_load_end - cpu_load_start:.2f} seconds")
print(f"Inference Time: {cpu_inference_end - cpu_inference_start:.2f} seconds")
print(f"Total Time: {cpu_inference_end - cpu_load_start:.2f} seconds")

# Store results
cpu_results = {
    'load_time': cpu_load_end - cpu_load_start,
    'inference_time': cpu_inference_end - cpu_inference_start,
    'total_time': cpu_inference_end - cpu_load_start,
    'output': output_cpu
}

# Clean up
del model_cpu
del tokenizer_cpu
gc.collect()

# ============================================================
# COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("BENCHMARK COMPARISON SUMMARY")
print("=" * 60)
print(f"\n{'Metric':<25} {'AirLLM (GPU)':<20} {'Transformers (CPU)':<20}")
print("-" * 65)
print(f"{'Model Load Time':<25} {airllm_results['load_time']:<20.2f} {cpu_results['load_time']:<20.2f}")
print(f"{'Inference Time':<25} {airllm_results['inference_time']:<20.2f} {cpu_results['inference_time']:<20.2f}")
print(f"{'Total Time':<25} {airllm_results['total_time']:<20.2f} {cpu_results['total_time']:<20.2f}")
print("-" * 65)

# Calculate speedup
if cpu_results['inference_time'] > 0:
    speedup = cpu_results['inference_time'] / airllm_results['inference_time']
    if speedup > 1:
        print(f"\n🚀 AirLLM is {speedup:.2f}x FASTER than CPU for inference!")
    else:
        print(f"\n🐢 CPU is {1/speedup:.2f}x FASTER than AirLLM for inference!")

print("\n" + "=" * 60)
