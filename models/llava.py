import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class Llava:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=0):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        """Loads the Llava model and processor."""
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor

    def generate_description(self, image, prompt, max_new_tokens=200):
        """Generates a detailed description of an image."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device, torch.float16)

        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        output = str(output).split('assistant')[1]
        if output.startswith("\n"):
            output = output[1:]  # Remove the first character
        return output

# Usage
if __name__ == "__main__":
    image = Image.open('dog.jpg').convert('RGB')
    llava = Llava()
    prompt = """
        Describe this image in as much detail as possible. 
        Identify the objects, as well as their relative positions. 
        Explain the scene, the actions taking place. 
        Include the setting, background details.
    """
    description = llava.generate_description(image, prompt)
    print(description)
