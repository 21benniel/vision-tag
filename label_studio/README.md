# Label Studio integration

VisionTag can export AI labels into [Label Studio](https://labelstud.io/) for human review.

## Prerequisites

```bash
pip install -e ".[label-studio]"
```

## Workflow

### 1. Auto-label images

```bash
visiontag run --images-dir ./my_images
```

### 2. Generate labeling interface

```bash
visiontag generate-label-studio-config -o label_studio_config.xml
```

Paste the XML into your Label Studio project under **Settings → Labeling Interface**.

### 3. Enable local file serving

```bash
visiontag fix-local-files --images-dir ./my_images
```

This writes Label Studio env vars so `/data/local-files/` URLs work. Restart Label Studio after running.

### 4. Build import JSON with predictions

```bash
visiontag build-label-studio-import labels.csv -o tasks_with_predictions.json \
  --images-dir ./my_images \
  --project-root .
```

### 5. Import and review

1. Start Label Studio: `label-studio`
2. Create or open a project
3. Enable **Use predictions to prelabel tasks** in project settings
4. Import `tasks_with_predictions.json`
5. Review each task, correct labels, and submit

## One-command pipeline

```bash
visiontag pipeline --images-dir ./my_images --limit 50
```

## Troubleshooting

### Images show 404 in Label Studio

- Run `visiontag fix-local-files` and restart Label Studio
- Ensure `--project-root` matches the directory Label Studio serves (`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`)
- Image URLs must be under that root

### Predictions not showing

- Turn on **Use predictions to prelabel tasks** in project settings
- Re-import `tasks_with_predictions.json` (predictions are embedded in the import file)

### Custom schema

Edit `config/label_schema.yaml`, then regenerate both the XML and import JSON.
