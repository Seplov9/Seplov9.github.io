from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             # {"type": "text", "text": "Describe this image."},
#             {"type": "text", "text": "How many images?"},
#         ],
#     }
# ]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 # "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
#                 ## 方案 A：指定每秒只取 1 帧（默认通常是 2 帧或更多）
#                 # "fps": 1.0, 
                
#                 ## 方案 B：强制整个视频只取固定的总帧数（比如只取 8 帧）
#                 # "nframes": 1000,
#                 "video": "./cat.mp4",
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "./cat.mp4",
            },
            # {
            #     "type": "video",
            #     "video": "./cat.mp4",
            # },
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
            {"type": "text", "text": "Describe what you see."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# image
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# video-1
# https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
# 需要在processing_qwen2_5_vl.py的"__call__""方法中，将fps = [metadata.sampled_fps for metadata in video_metadata]
# 修改为fps = output_kwargs["videos_kwargs"].get("fps", 2.0)

image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

if video_kwargs is not None:
    # video_kwargs["do_sample_frames"] = True
    video_kwargs["fps"] = video_kwargs["fps"][0]

# video_inputs[0] = video_inputs[0][:8]  # 强制只取前 8 帧（如果视频帧数超过 8）

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

# video-2
# https://github.com/huggingface/transformers/blob/a6dab9f359034b2745f787c5f4287747b8b62d30/docs/source/en/model_doc/qwen2_5_vl.md
# inputs = processor.apply_chat_template(
#     messages,
#     fps=2,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device)

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
