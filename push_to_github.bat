@echo off
setlocal
set REPO_URL=https://github.com/Pradyumanv68/retail-store-inventory-analytics-final.git
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Install it from https://git-scm.com/download/win , then double-click this file again.
  pause
  exit /b 1
)
if not exist .git git init
git config user.name >nul 2>nul || git config user.name "Pradyuman Verma"
git config user.email >nul 2>nul || git config user.email "Pradyumanv68@users.noreply.github.com"
git add .
git commit -m "Capstone: Retail Store Sales & Inventory Performance Analytics"
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%
echo.
echo A browser window may open asking you to sign in to GitHub. Please sign in and click Authorize.
git push -u origin main
if errorlevel 1 (
  echo.
  echo PUSH FAILED. Make sure you created an EMPTY public repo with the exact name and are signed in as Pradyumanv68.
  pause
  exit /b 1
)
echo.
echo SUCCESS! Open https://github.com/Pradyumanv68/retail-store-inventory-analytics-final
pause
