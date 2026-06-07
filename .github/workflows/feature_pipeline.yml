name: Feature Pipeline (hourly)

on:
  schedule:
    - cron: "0 * * * *"      # every hour
  workflow_dispatch:          # allows manual run from the Actions tab

permissions:
  contents: write             # needed to commit the updated feature store

jobs:
  update-features:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-pipeline.txt

      - name: Run feature pipeline
        run: python scripts/feature_pipeline.py

      - name: Commit updated feature store
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add feature_store.csv
          git commit -m "Update feature store [skip ci]" || echo "No changes to commit"
          git push
