# timegram estimator 7b3913dd18dfa3c5

```
select(probs=[0.2, 0.2, 0.2, 0.2, 0.2])
├── ar(1)
│   └── noise(normal(0, 1))
├── ar(2)
│   └── noise(normal(0, 1))
├── ar(3)
│   └── noise(normal(0, 1))
├── ar(4)
│   └── noise(normal(0, 1))
└── ar(5)
    └── noise(normal(0, 1))
observe: Missing(rate=0.1)
estimate: which()
loss: absolute (native scale)
```

- training: 30,000,000 simulated series, lengths (15, 100), batch 4096 (4096 processes x 1 replicates, accumulate 1), seed 0
- network: {'family': 'resmlp', 'width': 384, 'depth': 4, 'expansion': 2, 'dropout': 0.0}
- optimizer: {'family': 'muon_adamw', 'lr': 0.001, 'muon_lr': 0.01, 'momentum': 0.95, 'weight_decay': 0.01}
- final training loss: 0.8287
- timegram 0.1
- inputs are standardized per series and forecasts mapped back: predictions are location-scale equivariant (level of this world: fixed; see timegram.equivariance)
