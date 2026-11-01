# timegram

A grammar for constructing and exploring forecasting worlds. A *world* is a distribution over
time-series data-generating processes, written by composing sources, filters and combinators.
Paired with a *decision problem* (what is observed, what is estimated, the loss and the
objective), a world is turned into a reusable neural estimator by training on its simulations.

```python
import timegram as tg

world_hw = tg.ets("AAA", period=12)            # additive Holt-Winters, defaults for everything else
estimator = world_hw.pretrain()                # retrieved from the local library, or trained
result = estimator(y_hist)
print(result.forecast)

spiky = tg.noise(tg.mixture(tg.normal(), tg.normal(0, 10), weights=[.95, .05]))
world = (spiky >> tg.ar(1) >> world_hw) + (tg.noise(tg.student_t(3)) >> tg.ar(1))
problem = (world.observe(tg.missing(.1))
                .estimate(tg.forecast(h=24, quantiles=[.1, .5, .9]))
                .loss(tg.pinball(scale="relative"))
                .objective(tg.risk(weight=.1), tg.miscoverage(weight=.9, levels=[.9])))
print(world.typecheck())                       # established / refuted / unverified claims
estimator = problem.pretrain()
```

## Installation

Python 3.11 or later, PyTorch (a CUDA GPU for the full training budgets).

```bash
pip install -e ".[dev,validation,experiments]"
```

Trained estimators are stored in `$TIMEGRAM_HOME` (default `~/.cache/timegram`) and found again by
the identity of their decision problem.

## Trained networks

`pretrained/` holds the networks of the experiments; with `TIMEGRAM_HOME=pretrained` they are
found by name (`tg.library.pretrained("acc_r6_natural_forms")`), and the experiment scripts use
them without retraining:

| name | network |
|---|---|
| `acc_r6_natural_forms` | one forecaster for yearly, quarterly and monthly data (the selected world) |
| `acc_backbone` | the base world of the search |
| `acc_r8_intermittent` | the selected world with intermittent demand added |
| `tourcal_pinball`, `tourcal_control`, `tourcal_calibrated` | one-step quantiles for monthly series |
| `ar_order` | probabilities over the order of an AR(1)-AR(5) |
| `m1_mixhalf_minmax_50M_s0` | the additive exponential-smoothing world for M1 |

`notebooks/pretrained_networks.ipynb` ([open in Colab](https://colab.research.google.com/github/gyytpqzoad/timegram/blob/main/notebooks/pretrained_networks.ipynb); runs locally too, CPU is enough) uses them to
forecast M1, M3 and Tourism, forecast a new series, evaluate calibrated quantiles on monthly
Tourism and estimate the order of a short autoregression.

## Tests

```bash
pytest                     # unit and integration tests
pytest -m recovery         # long-series parameter recovery against statsmodels and arch
```

## Reproducing the experiments

Each script documents its commands at the top. Training runs on one GPU (6 GB is enough);
evaluation runs on a CPU. Results are written to `results/` next to each script.

**Data.** M1, M3, M4 and Tourism are read through `fcompdata`. The population records are
downloaded by `experiments/ar_order/ecology.py` (rgpdd, CC0). The intermittent-demand data are
exported from R: clone `fable.intermittent` into `experiments/intermittent/data/` and run
`Rscript export.R` there, and `Rscript carparts.R` for Car Parts (package `expsmooth`). The Monash
archive's Car Parts files (`car_parts_dataset_with_missing_values.tsf` and
`car_parts_dataset_without_missing_values.tsf`, from forecastingdata.org) go in the same folder.

### Reproducing a hand-written forecaster

```bash
python experiments/reproduce_backbone.py --seed 0      # quarterly M1, M3, M4, Tourism
```

### Grammar-guided exploration for forecasting accuracy

```bash
python experiments/accuracy/run.py train backbone                       # the base world
python experiments/accuracy/run.py train ft_backbone r6_natural_forms   # control and selected world
python experiments/accuracy/run.py validate backbone ft_backbone r6_natural_forms
python experiments/accuracy/run.py test backbone r6_natural_forms
python experiments/accuracy/chronos_test.py                             # Chronos-2 on the same windows
python experiments/accuracy/monash_table.py monash.tex                  # comparison with the Monash archive
```

Every candidate world of the search is defined in `CANDIDATES` in `run.py` and trained, validated
and tested with the same commands. The reference worlds with fitted priors need
`python experiments/accuracy/empirical_priors.py fit` and `summarize` first. The extension to
intermittent demand: `run.py train r8_intermittent`, then
`python experiments/accuracy/monash_counts.py r6_natural_forms r8_intermittent`.

### Per-series calibration on monthly Tourism

```bash
python experiments/tourism_calibration/run.py train pinball control calibrated
python experiments/tourism_calibration/run.py tourism test pinball:log control:log calibrated:log autoets chronos2
python experiments/tourism_calibration/run.py own pinball calibrated    # coverage by length, training world
python experiments/tourism_calibration/run.py report
```

The fine-tuning recipe was chosen on simulated AR(5) data: `experiments/ar5_calibration`.

### Order selection and the memory of animal populations

```bash
python experiments/ar_order/run.py train
python experiments/ar_order/run.py evaluate              # calibration in simulation
python experiments/ar_order/ecology.py                   # probabilities for the population records
python experiments/ar_order/ecology_predictive.py        # one-step predictive check
python experiments/ar_order/ecology_simple_figure.py
```

### Identification

```bash
python experiments/identification/sensitivity.py        # local sensitivity of two AR(1) components
```

### Intermittent demand

```bash
python experiments/intermittent/run.py train
python experiments/intermittent/search.py train                         # control and the single moves
python experiments/intermittent/search.py train persistent weekly_occ integer
python experiments/intermittent/search.py combine3                      # the moves that beat the control
python experiments/intermittent/search.py validate
python experiments/intermittent/search.py test --final   # benchmark windows and Car Parts (Monash protocol)
python experiments/intermittent/tweedie_gp_protocol.py   # Car Parts, protocol of Damato et al. (2025)
Rscript experiments/intermittent/baselines.R && python experiments/intermittent/score_r_baselines.py
```

### M1 with the additive ETS family

```bash
cd experiments/m1_ets && python run.py train mixhalf_minmax --budget 50000000 && python run.py report
```

## License

GPL-3.0-or-later.
