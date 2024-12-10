import torch
import numpy as np

from PIL import Image
from models import Llava
from models import Llama
from lang_sam import LangSAM
from lang_sam.utils import draw_image

def run_inference(image_pil: Image.Image, text_prompt, sam_type="sam2.1_hiera_small", box_threshold=0.3, text_threshold=0.25):
    # 初始化 LangSAM 模型
    model = LangSAM(sam_type=sam_type)

    # 模型推論
    print("Running inference...")
    results = model.predict(
        images_pil=[image_pil],
        texts_prompt=[text_prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]  # 只取第一個結果

    # 如果沒有檢測到任何物件，返回原始影像
    if results["masks"] is None or len(results["masks"]) == 0:
        print("No masks detected. Returning original image.")
        return image_pil


    # 繪製結果
    image_array = np.asarray(image_pil)
    output_image = draw_image(
        image_array,
        results["masks"],
        results["boxes"],
        results["scores"],
        results["labels"],
    )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

    return output_image

def main(inp, image_path):
    # load image
    image = Image.open(image_path).convert('RGB')

    # Generate image description with Llava
    llava = Llava()
    description_prompt = """
        Describe this image in as much detail as possible. 
        Identify the objects, as well as their relative positions. 
        Explain the scene, the actions taking place. 
        Include the setting, background details.
    """
    description = llava.generate_description(image, description_prompt)
    print('Output description:\n', description)

    # Extract all bojects mentioned in the description with Llama
    llama = Llama()
    question = f"Extract all objects mentioned in the description below: {description}"
    template = """
        You will get a description from an image, and you need to extract the objects mentioned in the image.
        For example: 
        Description: A dog is lying on the grass. 
        Answer: 1. dog, 2. grass.
        Now it's your turn: {question}
    """
    ans = llama.chat(question, template)
    print('Extracted objects:\n', ans)

    # Add descriptions for each extracted object with Llama
    objects = ans
    question = ""
    template = template = f"""
        Given the image description: "{description}", 
        provide concise and accurate descriptions of the following objects: {objects}.
        Focus on their visual characteristics and avoid redundant details.
    """

    ans = llama.chat(question, template)
    print('------------------------------------------------------------------------------------------------')
    print(ans)

    # Pick the things that relate to the input
    question = f"""Object list:
    {ans}
    Question or description: {inp}
    """
    template = template = f"""
        Based on the object list below and the given question or description, identify the objects that are directly related.
        If no objects are related, return "none."

        Example:
        Object list: 
            1. Oranges: Round, segmented citrus fruits with thick, rough skin and juicy pulp.
            2. Lemons: Small, spherical citrus fruits with thin, smooth skin and acidic juice-filled cavities.
            3. Strawberries: Fleshy, aggregate fruits with bright red color, white seeds, and a hollow core.
        Question or description: "strawberries"
        Answer: Strawberries.
        
        Now process the provided inputs.
        Object list: {ans}
        Question or description: "{inp}"
    """
    ##########################

    ans = llama.chat(question, template)
    print('------------------------------------------------------------------------------------------------')
    print(ans)

    # Extract object with Llama
    question = f"Extract all the objects mentioned below into a list: {ans}"
    template = """
        Extract all the objects mentioned in the input into a simple, comma-separated list.
        Example:
        Input:
            1. Fur: Light golden in color.
            2. Young golden retriever puppy: A small, light-golden canine with a joyful expression.
        Answer: Fur, Young golden retriever puppy.
        
        Now process the following input: {question}
    """
    ans = llama.chat(question, template)
    print('------------------------------------------------------------------------------------------------')
    print(ans)

    # Image segmentation with langsam
    output_path = "output_image.png"
    output_image = run_inference(image, ans)
    output_image.save(output_path)
    print(f"Inference completed. Output saved to {output_path}")

if __name__ == '__main__':
    while(True):
        inp = input('Prompt: ')
        image_path = input('Image path: ')
        main(inp, image_path)
