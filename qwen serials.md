# 关键文件
- transformers库  
transformers/src/transformers/processing_utils.py(class ProcessorMixin(PushToHubMixin))  
transformers/src/transformers/generation/utils.py(class GenerationMixin(ContinuousMixin))  
transformers/src/transformers/modeling_utils.py(class PreTrainedModel)  
transformers/tokenization_utils_base.py(class PreTrainedTokenizerBase)  
transformers/tokenization_utils_tokenizers.py(class TokenizersBackend)  
transformers/src/transformers/image_processing_utils.py  
transformers/src/transformers/video_processing_utils.py  
transformers/src/transformers/modeling_rope_utils.py  
transformers/src/transformers/models/qwen2_vl(v5.3.0 class Qwen2VLImageProcessorFast())  
transformers/src/transformers/models/qwen2/tokenization_qwen2.py  
transformers/src/transformers/models/auto/processing_auto.py  
transformers/src/transformers/models/auto/image_processing_auto.py  
transformers/src/transformers/models/auto/video_processing_auto.py  
transformers/docs/source/en/model_doc/qwen3_vl.md

- qwen_vl_utils库  
miniconda3/lib/python3.12/site-packages/qwen_vl_utils/vision_process.py(autodl)(qwen2vl qwen2.5vl)

- huggingface files  
config.json  
tokenizer.json  
preprocessor_config.json

- 模型参数  
transformers/src/transformers/models/qwen2_vl/processing_qwen2_vl.py  
transformers/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py  
transformers/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py  
transformers/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py  
huggingface config.json


# 对比
## processor
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
- transformers/models/qwen2_vl/image_processing_qwen2_vl.py  
  class Qwen2VLImageProcessor  
  `def _preprocess()`  
<img width="2673" height="1945" alt="image" src="https://github.com/user-attachments/assets/07fff3fc-3e17-41da-901f-9276b2054ea3" />

## modeling
### transformers/models/qwen3_vl/modeling_qwen3_vl.py
- 类结构
<img width="2154" height="1324" alt="image" src="https://github.com/user-attachments/assets/e9d35103-8f3d-4aa2-afd9-241d2f466f3c" />

- Attention
<img width="803" height="746" alt="image" src="https://github.com/user-attachments/assets/d04db389-e74a-4d4f-9e9a-293064b9f634" />
