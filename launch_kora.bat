@echo off
setlocal enabledelayedexpansion

REM Resolve project root as the directory of this script
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Ensure RAG and index folders exist
if not exist RAG mkdir RAG
if not exist .kora mkdir .kora
if not exist .kora\index mkdir .kora\index

REM Check UV availability
where uv >nul 2>nul
if errorlevel 1 (
  echo [KORA] 'uv' not found. Please install UV from https://docs.astral.sh/uv/ and re-run.
  exit /b 1
)

REM Install project and run
uv pip install -e .
uv run -m kora.launcher
