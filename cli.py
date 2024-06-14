from transformers import TextStreamer
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria
import requests
from io import BytesIO
import argparse
import warnings

warnings.filterwarnings("ignore")


DEFAULT_IMAGE_TOKEN = "<|image_1|>"

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def main(args):

    model_id = args.model_base
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=args.device, trust_remote_code=True, torch_dtype=torch.float16)

    if args.model_path:
        peft_model_id = args.model_path
        model.load_adapter(peft_model_id)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    messages = [
    # {"role": "system", "content": "You are an AI assistant monitoring traffic situations through surveillance systems to support drivers in emergency situations."},
    ]

    image = load_image(args.image_file)

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    while True:
        try:
            inp = input(f"User: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"Assistant: ", end="")

        if image is not None and len(messages) < 2:
            # only putting the image token in the first turn of user.
            # You could just uncomment the system messages or use it.
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        messages.append({"role": "user", "content": inp})

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to(args.device)
        
        stop_str = "<|end|>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, processor.tokenizer, inputs["input_ids"])
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, 
                streamer=streamer,
                **generation_args,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
                stopping_criteria=[stopping_criteria]
            )

        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        messages.append({"role":"assistant", "content": outputs})

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)