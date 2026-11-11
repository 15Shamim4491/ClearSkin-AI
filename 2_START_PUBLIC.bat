@echo off
title Acne Detector - Public Link
cd /d "%~dp0backend"
echo Flask server chalu hocche...
start "Flask Server (eta bondho korba na)" cmd /k python app.py
echo Server ready hoar jonno 6 second opekkha...
timeout /t 6 > nul
echo.
echo ================================================================
echo  Niche ekta link ashbe: https://xxxx.trycloudflare.com
echo  Oita copy kore browser-e kholo ba bondhuke pathao.
echo  Ei window ar Flask window duto khola rakho.
echo ================================================================
echo.
cloudflared tunnel --url http://localhost:5000
pause
