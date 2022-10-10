# LightweightLearnedInertialOdometer
Code for paper: "LLIO: Lightweight Learned Inertial Odometer"

## Prerequisites

Install dependency use pip:
```bash
pip install torch einops numpy
```

## Usage
The LLIO contained in the model_twolayer.py, and a unit test for illustration input and output are provided in this file.
````python
model_para = {
        "input_len": 100,
        "input_channel": 6,
        "patch_len": 25,
        "feature_dim": 512,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": { # include: Feature Convert & ResMLP Module in the paper Fig. 3.
            "name": "ResMLP",
            "layer_num": 6,
            "expansion": 2,
            "dropout": 0.2,
        },
        "reg": { # Regression in the paper Fig.3
            "name": "MeanMLP",
            # "name": "MaxMLP",
            "layer_num": 3,
        }
    }

    net = TwoLayerModel(model_para) # initialize the model
    x = torch.rand([512, 6, 100]) # batch_size, input_channel, input_len,

    y, y_cov = net(x) # output: [batch_size, 3], [batch_size, 3]
    print('x:', x.shape, 'y', y.shape, 'y_cov', y_cov.shape)
````

Example output:
```bash
x: torch.Size([512, 6, 100]) y torch.Size([512, 3]) y_cov torch.Size([512, 3])
```

## Acknowledgements
Thanks for TLIO [https://github.com/CathIAS/TLIO].

## License
The source code is released under GPLv3 license.