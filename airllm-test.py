from airllm import AutoModel
import time

MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")

input_text = [
    'Hello, how are you?',
]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)

# Start timing before inference
start_time = time.time()

generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=10,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

# End timing after output is ready
end_time = time.time()

print(output)
print(f"Inference time: {end_time - start_time:.2f} seconds ---")
