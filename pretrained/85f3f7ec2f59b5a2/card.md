# timegram estimator 85f3f7ec2f59b5a2

```
select(probs=[0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143])
├── arima(3, 1, 3, prior=sarima_tree(period=1))
│   └── noise(normal(0, 1))
├── arima(3, 1, 3, seasonal=(2, 1, 2), period=4, prior=sarima_tree(period=4))
│   └── noise(normal(0, 1))
├── arima(3, 1, 3, seasonal=(2, 1, 2), period=12, prior=sarima_tree(period=12))
│   └── noise(normal(0, 1))
├── ets(ANN)
│   └── noise(normal(0, 1))
├── ets(AAN)
│   └── noise(normal(0, 1))
├── ets(AAA, period=4)
│   └── noise(normal(0, 1))
└── ets(AAA, period=12)
    └── noise(normal(0, 1))
observe: Missing(rate=0.1)
estimate: forecast(h=24)
loss: absolute (relative scale)
```

- training: 150,000,000 simulated series, lengths (7, 128), batch 2048 (2048 processes x 1 replicates, accumulate 1), seed 0
- network: {'family': 'resmlp', 'width': 384, 'depth': 4, 'expansion': 2, 'dropout': 0.0}
- optimizer: {'family': 'muon_adamw', 'lr': 0.001, 'muon_lr': 0.01, 'momentum': 0.95, 'weight_decay': 0.01}
- final training loss: 0.6831
- timegram 0.1
- inputs are standardized per series and forecasts mapped back: predictions are location-scale equivariant (level of this world: partly; see timegram.equivariance)
