@echo off
echo Installing dependencies for Model Training...

:: Install PyTorch with CUDA support (assuming CUDA 12.1, adjust if needed)
py -3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install Unsloth (optimized training library)
py -3.10 -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

:: Install other requirements
:: We install bitsandbytes-windows for Windows compatibility if not on WSL
py -3.10 -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
py -3.10 -m pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate

echo.
echo Setup complete!
pause
