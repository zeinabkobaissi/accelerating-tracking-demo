export DAGSHUB_USER="zeinabkoubaissi33"
export DAGSHUB_TOKEN="xxx"

docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=" https://dagshub.com/zeinabkoubaissi33/accelerating-tracking-demo.mlflow" \
  -e MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER" \
  -e MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN" \
  -e DAGSHUB_OWNER="zeinabkoubaissi33" \
  -e DAGSHUB_REPO="accelerator-tracking-demo" \
  tracking-demos:latest