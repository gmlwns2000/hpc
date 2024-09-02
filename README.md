# HPC test

Playground of Heejun for Intel Gaudi.

## Measured TFLOP/s

| Device    | Linear(fp32)  | Linear(bf16)  | Linear(fp16)  | Conv(fp32)    | Conv(bf16)    | Conv(fp16)    |
|-----------|---------------|---------------|---------------|---------------|---------------|---------------|
| RTX4090   | 46.26         | 147.79        | 145.41        | 58.47         | 117.56        | 116.57        |
| A100-80   | 17.34         | 228.21        | 211.80        | 99.13         | 182.58        | 179.20        |
| Gaudi-2   | 69.70         | 308.97        | 287.11        | 56.65         | 139.94        | 136.49        | 

