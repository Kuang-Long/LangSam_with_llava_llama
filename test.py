import numpy as np

from PIL import Image
from models import Llava
from models import Llama
from models import QuestionDetector
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

    # initialize models
    llava = Llava()
    llama = Llama()
    qd = QuestionDetector()

    # Generate image description with Llava
    description = llava.chat(image, prompt=inp)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Output description:\n', description)

    is_question = qd.is_question(inp)
    ans = llava.chat(image, prompt=inp, is_question=is_question)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(ans)
    if not is_question:
        question = f"{ans}"
        template = "what are the main objects explicitly described in this: {question}"
        ans = llama.chat(question, template)
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('llama: ' + ans)

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
