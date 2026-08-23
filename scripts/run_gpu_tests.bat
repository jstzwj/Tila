@echo off
rem Windows 下运行 GPU 集成测试：triton 后端需要 MSVC 环境 + CC=cl
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
set CC=cl
set PYTHONPATH=G:\github\Tila\src
cd /d G:\github\Tila
python -m pytest tests/test_gpu_integration.py -q
