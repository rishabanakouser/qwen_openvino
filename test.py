import openvino_genai
import openvino as ov
import numpy as np
from PIL import Image

MODEL_PATH = 'models/Qwen2.5-VL-7B-Instruct-int8-ov'

print('Loading Qwen2.5-VL OpenVINO model using VLMPipeline...')
pipe = openvino_genai.VLMPipeline(MODEL_PATH, 'GPU')
print('SUCCESS: Pipeline loaded.')

img = Image.new('RGB', (100, 100))
img_tensor = ov.Tensor(np.array(img))

prompt = 'Describe this image.'

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 20

res = pipe.generate(prompt, images=[img_tensor], generation_config=config)
print('Generated:', res.texts)
