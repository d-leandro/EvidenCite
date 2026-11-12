@echo off
setlocal
REM Configura el remoto DVC (Google Drive) en .dvc\config.local
REM Requiere las variables ID_CARPETA, GDRIVE_CLIENT_ID y GDRIVE_CLIENT_SECRET

cd /d "%~dp0"

if not defined ID_CARPETA goto :faltan
if not defined GDRIVE_CLIENT_ID goto :faltan
if not defined GDRIVE_CLIENT_SECRET goto :faltan

uv run dvc remote add --local -d -f gdrive-evidencite "gdrive://%ID_CARPETA%"
if errorlevel 1 goto :error
uv run dvc remote modify --local gdrive-evidencite gdrive_client_id "%GDRIVE_CLIENT_ID%"
if errorlevel 1 goto :error
uv run dvc remote modify --local gdrive-evidencite gdrive_client_secret "%GDRIVE_CLIENT_SECRET%"
if errorlevel 1 goto :error

echo Remoto DVC configurado. Ahora ejecuta: uv run dvc pull
exit /b 0

:faltan
echo Error: define ID_CARPETA, GDRIVE_CLIENT_ID y GDRIVE_CLIENT_SECRET antes de ejecutar este script.
exit /b 1

:error
echo Error al configurar el remoto DVC.
exit /b 1
