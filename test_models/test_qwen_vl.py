from PIL import Image
import os
import shutil
import uuid
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

torch.set_default_dtype(torch.bfloat16)
# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    # "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# processor = AutoProcessor.from_pretrained("Ertugrul/Qwen2-VL-7B-Captioner-Relaxed")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

folder = "Camera"

for entry in os.scandir(folder):
    if entry.name.endswith(".txt"):
        continue

    image = Image.open(entry.path).convert('RGB')
    image.thumbnail((1280, 1280))

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=300)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    
    name = entry.name.rsplit(".", 1)[0]
    with open(os.path.join(folder, f"{name}.txt"), "w") as f:
        f.write(output_text+"\n")
    
    print(output_text)
    print()
    # if output_text.startswith("Yes"):
        # shutil.copy(entry.path, os.path.join("selected", entry.name))