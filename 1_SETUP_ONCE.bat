@echo off
title Setup (ekbar korlei hobe)
echo ===== 1/2: Python package install hocche =====
cd /d "%~dp0backend"
python -m pip install -r requirements.txt
echo.
echo ===== 2/2: cloudflared install hocche =====
where cloudflared >nul 2>nul
if %errorlevel%==0 (
  echo cloudflared agei install kora ache.
) else (
  winget install --id Cloudflare.cloudflared -e --accept-package-agreements --accept-source-agreements
)
echo.
echo Setup shesh! Ekhon VS Code / CMD bondho kore abar khulo, tarpor 2_START_PUBLIC.bat chalao.
pause
