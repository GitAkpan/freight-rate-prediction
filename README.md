# Freight Rate Prediction

## Run
```bash
python -m pip install -r requirements.txt
# place train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv in data/
python train.py      # time-based CV, final fit, writes validation_predictions.csv and fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
Outputs: `validation_predictions.csv`, `cv_results.csv`, `scorer_results/candidate_december.png`.

## Approach
- Target: log(posted_rate); model: HistGradientBoostingRegressor.
- Features: log distance, |weight|, equipment, pickup, delivery, day of week, quote_signal.
- Cleaning: absolute weight, two-pass removal of corrupted rates (1.4% of rows).
- Validation: 3 rolling-origin time splits (train on past, test next two months).
