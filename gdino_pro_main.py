from PIL import Image
from models import Llava
from models import Llama
from models import QuestionDetector
from lang_sam import BoundingBoxSAM

def main(inp, image_path, image_url):
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
        template = "what are the main objects explicitly described in this: {question} and related to {inp}"
        ans = llama.chat(question, template)
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('llama: ' + ans)

    # Image segmentation with langsam
    bbox_sam = BoundingBoxSAM(token=token)
    bbox_sam.process_image(image_url, image_path, ans, output_path='output.jpg')

# Example usage
if __name__ == "__main__":
    # use ur token here
    token = "token" 
    while(True):
        inp = input('Prompt: ')
        image_path = input('Image path: ')
        image_url = input('url: ')
        main(inp, image_path, image_url)