@echo off
setlocal

rem Run this batch file inside the folder where you want the project created.

echo Creating project folders...
for %%D in (
    "data"
    "model"
    "app"
    "notebooks"
    ".github"
    "airflow"
    "monitoring"
) do (
    if not exist "%%~D" mkdir "%%~D"
)

echo Creating project files...
for %%F in (
    "requirements.txt"
    "Dockerfile"
    "deployment.yaml"
    "service.yaml"
    "train.py"
    "preprocess.py"
    "app.py"
    "dvc.yaml"
    "README.md"
) do (
    if not exist "%%~F" type nul > "%%~F"
)

echo.
echo Project structure created successfully.
endlocal
