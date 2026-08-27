@echo off
rem Windows 下运行 oracle 第三轮（v0.6 attention 布局实验，docs/v0.6-attention.md §8.1）
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
set CC=cl
set PYTHONPATH=G:\github\Tila\src
cd /d G:\github\Tila
python tools\oracle_round3_attention.py %*
