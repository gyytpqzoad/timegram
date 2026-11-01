# timegram estimator 99f8e2b28ffce673

```
select(probs=[0.5, 0.5])
├── ets(AAA, period=12)
│   └── noise(normal(0, 1))
└── ets(ANA, period=12)
    └── noise(normal(0, 1))
observe: complete
estimate: forecast(h=1, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
loss: pinball (relative scale)
```

- training: 81,920,000 simulated series, lengths (24, 96), batch 4096 (128 processes x 32 replicates, accumulate 1), seed 0
- network: {'family': 'resmlp', 'width': 256, 'depth': 4, 'expansion': 2, 'dropout': 0.0}
- optimizer: {'family': 'adamw', 'lr': 0.001, 'weight_decay': 0.01}
- final training loss: 0.0979
- timegram 0.1
- inputs are standardized per series and forecasts mapped back: predictions are location-scale equivariant (level of this world: free; see timegram.equivariance)
