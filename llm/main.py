import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

colors_list = [
    "102;194;165",
    "252;141;98",
    "141;160;203",
    "231;138;195",
    "166;216;84",
    "255;217;47",
]


def load():
    # name = "/home/vernica/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85/"
    # name = "/home/vernica/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/"
    # name = "/home/vernica/.cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots/11fb331c2c388748c110926aa8013161cb5a85b5/"

    # name = "/home/vernica/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/"

    name = "microsoft/Phi-3-mini-4k-instruct"
    # name = "sentence-transformers/all-MiniLM-L6-v2"
    # name = "gpt-2"

    name_save = "phi-3-4b"

    # Configure 4-bit quantization
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     # bnb_4bit_compute_dtype=torch.float16,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_use_double_quant=True,
    # )

    # Load model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(
    #     name,
    #     quantization_config=quantization_config,
    #     device_map="cuda",  # Use GPU if available
    #     # device_map="auto",  # Automatically determine device mapping
    # )
    # model.save_pretrained(name_save)

    model = AutoModelForCausalLM.from_pretrained(name_save)
    tokenizer = AutoTokenizer.from_pretrained(name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        # do_sample=False,
    )
    return model, tokenizer, generator


def show_tokens(sentence=None, name=None, token_ids=None, tokenizer=None):
    if token_ids is None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(name)
        token_ids = tokenizer(sentence).input_ids
    for idx, t in enumerate(token_ids):
        print(
            f"\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m"
            + tokenizer.decode(t)
            + "\x1b[0m",
            end=" ",
        )
    print()


model, tokenizer, generator = load()
# print(model)

# prompt = [{"role": "user", "content": "Create a funny joke about chickens."}]
# prompt = [
#     "<|system|>",
#     "You are a helpful assistant.<|end|>",
#     "<|user|>",
#     "How to explain Internet for a medieval knight?<|end|>",
#     "<|assistant|>",
# ]
# prompt = "Create a funny joke about chickens."
# prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"
# prompt = "Write an email apologizing to Sarah"
prompt = "The capital of Spain is"

# show_tokens(messages, tokenizer=tokenizer)

# output = generator(messages)
# print(output[0]["generated_text"])

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
print("Input")
show_tokens(tokenizer=tokenizer, token_ids=input_ids[0])

model_output = model.model(input_ids)
head_output = model.lm_head(model_output[0])

token_ids = [head_output[0, -1].argmax(-1)]
print("Output")
show_tokens(tokenizer=tokenizer, token_ids=token_ids)

# generation_output = model.generate(input_ids=input_ids, max_new_tokens=20)

# print(tokenizer.decode(generation_output[0]))

# for id in generation_output[0]:
#     print(id, tokenizer.decode(id))
