CAPTIONING_MODEL_NAMES = {
    "git-base-coco": ("microsoft/git-base-coco", "AutoModelForCausalLM"),
    "git-large-coco": ("microsoft/git-large-coco", "AutoModelForCausalLM"),
    "blip-image-captioning-base": ("Salesforce/blip-image-captioning-base", "BlipForConditionalGeneration"),
    "blip-image-captioning-large": ("Salesforce/blip-image-captioning-large", "BlipForConditionalGeneration"),
    "vit-gpt2-image-captioning": ("nlpconnect/vit-gpt2-image-captioning", "VisionEncoderDecoderModel"),
    "Qwen2-VL-7B-Instruct-GPTQ-Int4": ("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", "Qwen2VLForConditionalGeneration")
}

TRANSLATION_MODEL_NAMES = {
    "mbart-large-50-many-to-many-mmt": ("facebook/mbart-large-50-many-to-many-mmt", "MBartForConditionalGeneration")
}

SEGMENTATION_MODEL_NAMES = {
    "Florence-2-large-ft": ("microsoft/Florence-2-large-ft", "AutoModelForCausalLM"),
    "Florence-2-large": ("microsoft/Florence-2-large", "AutoModelForCausalLM"),
    "Florence-2-base-ft": ("microsoft/Florence-2-base-ft", "AutoModelForCausalLM"),
    "Florence-2-base": ("microsoft/Florence-2-base", "AutoModelForCausalLM")
}