# Transformer Profiling Results 

## Setup
- Google Colab G4 GPU (90GB RAM)
- BS=2, VOCAB=10000, CTX_LEN=1024
- Exclude XL since it can not be fit into G4

## Results
### With Warm-up

|        |   forward_min |   forward_mean |   forward_stdev |   backward_min |   backward_mean |   backward_stdev |   optimizer_min |   optimizer_mean |   optimizer_stdev |
|:-------|--------------:|---------------:|----------------:|---------------:|----------------:|-----------------:|----------------:|-----------------:|------------------:|
| small  |     0.0238351 |      0.0244138 |     0.000262148 |      0.0511546 |       0.0570904 |      0.0169245   |      0.00755261 |        0.0100707 |       0.00789251  |
| medium |     0.0718396 |      0.0723415 |     0.000329448 |      0.144306  |       0.144783  |      0.000519937 |      0.024142   |        0.024435  |       0.000749599 |
| large  |     0.149776  |      0.150075  |     0.000517121 |      0.299364  |       0.299659  |      0.000125234 |      0.0548038  |        0.0553655 |       0.0014733   |
| xl     |     0.371067  |      0.371247  |     0.000405548 |      0.676559  |       0.6772    |      0.00035601  |      0.189041   |        0.191016  |       0.00559393  |

### Without warm-up
|        |   forward_min |   forward_mean |   forward_stdev |   backward_min |   backward_mean |   backward_stdev |   optimizer_min |   optimizer_mean |   optimizer_stdev |
|:-------|--------------:|---------------:|----------------:|---------------:|----------------:|-----------------:|----------------:|-----------------:|------------------:|
| small  |     0.0244019 |      0.0375004 |     0.041109    |      0.0513975 |        0.057439 |      0.0170415   |      0.00754896 |        0.010113  |       0.00792517  |
| medium |     0.0721415 |      0.0729764 |     0.000727506 |      0.144412  |        0.145303 |      0.00107894  |      0.0241334  |        0.0244452 |       0.000787924 |
| large  |     0.150004  |      0.150492  |     0.000717356 |      0.299414  |        0.299774 |      0.000330989 |      0.0546608  |        0.0552555 |       0.00144742  |
| xl     |     0.370839  |      0.371132  |     0.000560507 |      0.676416  |        0.677299 |      0.000503894 |      0.189397   |        0.191315  |       0.0054042   |

## Analysis

Forward pass, comparing **with → without** warm-up:

| Model  | forward_mean (with → without) | forward_stdev (with → without) |
| ------ | ----------------------------- | ------------------------------ |
| small  | 0.0244 → 0.0375 (+54%)        | 0.00026 → 0.0411 (~150×)       |
| medium | 0.0723 → 0.0730               | 0.00033 → 0.00073              |
| large  | 0.1501 → 0.1505               | 0.00052 → 0.00072              |
| xl     | 0.3712 → 0.3711               | 0.00041 → 0.00056              |

It is impacted most strongly, compared to backward and optimizer step. We argue that it is because forward step takes place at every beginning of each train iteration. 

Besides, it is notable that small model suffers from the latency of preloading after removing warmup stages.