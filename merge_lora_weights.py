import argparse
from utils import get_model_name_from_path, load_pretrained_model

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    # I get an shared memmory error between embed_token and vision_embed_token.wte
    # I don't know why this error occurs, So I just save the model with out safe_serialization
    # If you know the reason, please let me know.
    model.save_pretrained(args.save_model_path, safe_serialization=False)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)