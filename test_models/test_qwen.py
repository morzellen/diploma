import json
from transformers import AutoModelForCausalLM, AutoTokenizer

captioning_model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    captioning_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(captioning_model_name)

# prompt = json.dumps({
#     "prompt": 'image description is: "woman holding a cat". Select categories for a photo from a list: ["people", "cars", "animals", "architecture", "landscape", "food", "other"]. Output answer as JSON.',
#     "format": "json",
#     "answer_format": {"category": []}
# })

prompt = "Напиши сочинение о картине мишки в лесу"

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# answer = json.loads(response)
# print(answer)
# print(answer["category"])