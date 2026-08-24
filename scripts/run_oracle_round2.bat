@echo off
rem Windows 下运行 oracle 第二轮（v0.5 归约布局实验，docs/v0.5-reduce.md §8.1）
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
set CC=cl
set PYTHONPATH=G:\github\Tila\src
cd /d G:\github\Tila
python tools\oracle_round2_reduce.py %*
