uv run dvc remote add --local -d gdrive-evidencite "gdrive://$ID_CARPETA"
uv run dvc remote modify --local gdrive-evidencite gdrive_client_id "$GDRIVE_CLIENT_ID"
uv run dvc remote modify --local gdrive-evidencite gdrive_client_secret "$GDRIVE_CLIENT_SECRET"