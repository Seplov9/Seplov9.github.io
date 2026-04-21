# 关键文件
transformers库  
transformers/src/transformers/processing_utils.py(class ProcessorMixin(PushToHubMixin))  
transformers/src/transformers/image_processing_utils.py  
transformers/src/transformers/video_processing_utils.py  
transformers/src/transformers/models/qwen2_vl(v5.3.0 class Qwen2VLImageProcessorFast(BaseImageProcessorFast))  
transformers/src/transformers/models/auto/processing_auto.py  
transformers/src/transformers/models/auto/image_processing_auto.py  
transformers/src/transformers/models/auto/video_processing_auto.py  
transformers/docs/source/en/model_doc/qwen3_vl.md

qwen_vl_utils库  
miniconda3/lib/python3.12/site-packages/qwen_vl_utils/vision_process.py(autodl)

huggingface files  
config.json  
tokenizer.json  
preprocessor_config.json

# 对比
## processing
<img width="1043" height="138" alt="image" src="https://github.com/user-attachments/assets/c48e9cb5-0683-41c0-9fb3-f18e7e7cbe3b" />

```python
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
image_processor = processor.image_processor
video_processor = processor.video_processor
<class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>
<class 'transformers.models.qwen2_vl.video_processing_qwen2_vl.Qwen2VLVideoProcessor'>

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
<class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>
<class 'transformers.models.qwen2_vl.video_processing_qwen2_vl.Qwen2VLVideoProcessor'>

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
<class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>
<class 'transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessor'>
```

## video_processing

## image_processing

# qwen-3vl
## modeling_qwen3_vl.py
<img width="803" height="746" alt="image" src="https://github.com/user-attachments/assets/d04db389-e74a-4d4f-9e9a-293064b9f634" />
