# timegram estimator 6772db71cafd58cc

```
select(probs=[0.5, 0.5])
├── ets(AAA, period=12)
│   └── noise(normal(0, 1))
└── ets(ANA, period=12)
    └── noise(normal(0, 1))
observe: complete
estimate: forecast(h=1, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
loss: pinball (relative scale)
objective: risk(weight=0.1, over=mean) + miscoverage(weight=0.9, over=mean, levels=(0.9,), smoothing='spread')
```

- training: 20,000,000 simulated series, lengths (24, 96), batch 4096 (32 processes x 128 replicates, accumulate 1), seed 3, from 99f8e2b28ffce673
- network: {'family': 'resmlp', 'width': 256, 'depth': 4, 'expansion': 2, 'dropout': 0.0}
- optimizer: {'family': 'adamw', 'lr': 0.0001, 'weight_decay': 0.01}
- final training loss: 0.0099
- timegram 0.1
- inputs are standardized per series and forecasts mapped back: predictions are location-scale equivariant (level of this world: free; see timegram.equivariance)
