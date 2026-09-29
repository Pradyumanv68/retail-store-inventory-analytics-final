#!/bin/sh
REPO_URL="${REPO_URL:-https://github.com/Pradyumanv68/Retail-Store-Sales-Inventory-Performance-Analytics.git}"
cd "$(dirname "$0")" || exit 1
command -v git >/dev/null || { echo "Install git first"; exit 1; }
[ -d .git ] || git init
git config user.name >/dev/null || git config user.name "Pradyuman Verma"
git config user.email >/dev/null || git config user.email "Pradyumanv68@users.noreply.github.com"
git add .
git commit -m "Capstone: Retail Store Sales & Inventory Performance Analytics"
git branch -M main
git remote remove origin 2>/dev/null
git remote add origin "$REPO_URL"
git push -u origin main && echo "SUCCESS" || echo "PUSH FAILED - check repo name and login"
