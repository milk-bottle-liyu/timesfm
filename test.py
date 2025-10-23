import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("/data/yup/personal/models/google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=512,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=False,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100) * 100,
#        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
print(point_forecast.shape)  # (2, 12)
print(point_forecast)
vec_list = model.encode(inputs=[
        np.linspace(0, 1, 100) * 100,
    ])
print(vec_list[0].shape)

# print(quantile_forecast.shape)  # (2, 12, 10): mean, then 10th to 90th quantiles.
# print(quantile_forecast)
