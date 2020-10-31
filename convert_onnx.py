import torchreid

torchreid.models.show_avai_models()

model = torchreid.models.build_model(name='osnet_x0_25', num_classes=1041)
torchreid.utils.load_pretrained_weights(model, "log/osnet_x0_25_market1501_softmax/model/model.pth.tar-180") 

model.eval()

from torch.autograd import Variable
import torch
import onnx

# An example input you would normally provide to your model's forward() method.
input = torch.ones(1, 3, 96, 48)
raw_output = model(input)

torch.onnx.export(model, input, 'osnet_ain_x1_0.onnx', verbose=False, export_params=True)
