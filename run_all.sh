#!/usr/bin/env bash
# One-command setup: trains models, runs experiments, migrates, seeds demo data.
set -e
python ml/classification/train.py
python ml/similarity/duplicates.py
python ml/forecasting/forecast.py
python ml/eda/run_eda.py
python ga/experiments/run_comparison.py 30
python ga/experiments/make_plots.py
python manage.py migrate
python manage.py seed_demo --reset
python manage.py test tests
echo "Done. Start the app with: python manage.py runserver"
