# video

def _prepare_position_ids_for_generation()
(Pdb) position_ids.shape
torch.Size([1, 13430])


def prepare_inputs_for_generation()
(Pdb) pp model_inputs
{'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], device='cuda:0'),
 'image_grid_thw': None,
 'input_ids': tensor([[151644,    872,    198,  ..., 151644,  77091,    198]],
       device='cuda:0'),
 'logits_to_keep': 1,
 'mm_token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0]], device='cuda:0'),
 'past_key_values': DynamicCache(layers=[DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer]),
 'pixel_values': None,
 'pixel_values_videos': tensor([[-1.0000, -1.0000, -1.0000,  ..., -0.5686, -0.5608, -0.5608],
        [-1.0000, -1.0000, -1.0000,  ..., -0.5373, -0.5451, -0.5451],
        [-1.0000, -1.0000, -1.0000,  ..., -0.9373, -0.9373, -0.9451],
        ...,
        [-1.0000, -1.0000, -1.0000,  ..., -1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000,  ..., -1.0000, -1.0000, -1.0000],
        [-1.0000, -1.0000, -1.0000,  ..., -1.0000, -1.0000, -1.0000]],
       device='cuda:0'),
 'position_ids': tensor([[[    0,     1,     2,  ..., 13427, 13428, 13429]],

        [[    0,     1,     2,  ...,  3637,  3638,  3639]],

        [[    0,     1,     2,  ...,  3637,  3638,  3639]],

        [[    0,     1,     2,  ...,  3637,  3638,  3639]]], device='cuda:0'),
 # position_ids.shape = [4, 1, 13430]
 'use_cache': True,
 'video_grid_thw': tensor([[178,  12,  22]], device='cuda:0')}
