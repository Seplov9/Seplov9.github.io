class GenerationMixin(ContinuousMixin):


    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """
        Tries to infer position ids given attention mask and past kv cache length. All instances when
        `position_ids=None` should call this method.
        """
        # `input_ids` may be present in the model kwargs, instead of being the main input (e.g. multimodal model)

        '''
        (Pdb) pp model_kwargs
        {'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], device='cuda:0'),
         'image_grid_thw': tensor([[  1,  86, 128]], device='cuda:0'),
         'mm_token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0]], device='cuda:0'),
         'pixel_values': tensor([[ 0.4196,  0.4196,  0.4275,  ...,  0.5922,  0.5922,  0.5922],
                [ 0.4667,  0.4667,  0.4667,  ...,  0.6235,  0.6235,  0.6235],
                [ 0.4667,  0.4667,  0.4745,  ...,  0.6078,  0.6157,  0.6157],
                ...,
                [-0.1529, -0.1608, -0.1608,  ..., -0.3255, -0.3176, -0.3176],
                [-0.2078, -0.2078, -0.2078,  ..., -0.3333, -0.3412, -0.3490],
                [-0.1765, -0.2000, -0.2235,  ..., -0.4196, -0.4275, -0.4353]],
               device='cuda:0')}
        (Pdb) inputs_tensor.shape
        torch.Size([1, 2766])
        (Pdb) inputs_tensor  # input_ids
        tensor([[151644, 872, 198, ..., 151644, 77091, 198]], device='cuda:0')
        '''
      
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:  # False
            inputs_tensor = model_kwargs["input_ids"]

        seq_length = inputs_tensor.shape[1]

        # 1. 尝试从参数中获取 attention_mask，并使用海象运算符 (:=) 直接赋值。
        #    如果掩码存在（不为 None），则进入位置 ID 的手动计算逻辑。
        if (attention_mask := model_kwargs.get("attention_mask")) is not None:
            
            # 2. 计算位置索引：
            #    - .long(): 将掩码（通常是 0 或 1）转为长整型。
            #    - .cumsum(-1): 在最后一个维度上计算累加。
            #      例如：mask 为 [1, 1, 0, 1] -> 累加后为 [1, 2, 2, 3]
            #    - - 1: 将序号转为从 0 开始的索引。
            #      接上例：[1, 2, 2, 3] - 1 -> [0, 1, 1, 2]
            position_ids = attention_mask.long().cumsum(-1) - 1
            '''
            (Pdb) position_ids.shape
            torch.Size([1, 2766])
            (Pdb) position_ids
            tensor([[   0,    1,    2,  ..., 2763, 2764, 2765]], device='cuda:0')
            '''
            
            # 3. 修正填充位（Padding Tokens）的位置：
            #    如果不处理，原先 mask 为 0 的位置在减 1 后会变成 -1（如上例所示）。
            #    这里使用 .masked_fill 将所有 attention_mask 为 0 的位置强制重置为 0。
            #    虽然这些位置会被 mask 掉，但保证 position_ids 合法能避免某些 Embedding 层报错。
            #    接上例：[0, 1, 1, 2] -> 处理后 [0, 1, 0, 2] （其中索引 2 的位置是 0，表示它是 Padding）
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            '''
            (Pdb) position_ids.shape
            torch.Size([1, 2766])
            (Pdb) position_ids
            tensor([[   0,    1,    2,  ..., 2763, 2764, 2765]], device='cuda:0')
            '''
          
        else:
            past_length = 0
            if (cache := model_kwargs.get("past_key_values")) is not None:
                past_length = cache.get_seq_length()

            position_ids = torch.arange(seq_length + past_length, dtype=torch.long, device=inputs_tensor.device)
            position_ids = position_ids.unsqueeze(0)
        return position_ids





    def generation(){
    
        # 1. 检查当前参数包 model_kwargs 中是否已经包含了有效的 position_ids
        #    使用 .get() 避免键不存在时抛出 KeyError，is not None 确保该值不是空对象
        kwargs_has_position_ids = model_kwargs.get("position_ids", None) is not None
        
        # 2. 检查当前模型实例的 forward 方法是否支持接收名为 "position_ids" 的参数
        #    通过 inspect.signature 获取函数签名，并将其参数名转为集合进行快速匹配
        accepts_position_ids = "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        
        # 3. 自动补全逻辑：
        #    条件 A: 用户/上游函数没有显式提供 position_ids
        #    条件 B: 模型本身支持接收 position_ids 参数
        #    条件 C: 当前模型不是 编码器-解码器(Encoder-Decoder) 结构（如 T5/BART，其位置处理逻辑不同）
        if not kwargs_has_position_ids and accepts_position_ids and not self.config.is_encoder_decoder:
            # 如果满足上述条件，调用内部方法根据 inputs_tensor 的维度和缓存状态
            # 自动计算并生成当前推理步所需的位置 ID，并存入参数字典中
            model_kwargs["position_ids"] = self._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
