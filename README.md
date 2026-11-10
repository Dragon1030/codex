# Research projects

This repository contains two independent projects:

## ContinualSteg

`ContinualSteg/` is a continual-learning baseline for robust image hiding. It
uses task adapters, prompts, replay, and distillation with DIV2K or another
unlabelled image folder.

See [ContinualSteg/README.md](ContinualSteg/README.md) for server, Docker,
training, resume, and evaluation commands.

## GraduationProject

`GraduationProject/` is a desktop Tkinter application for ping-pong beginner
action assessment. It uses OpenCV and MediaPipe to analyze local videos and
stores user and analysis records in SQLite.

See [GraduationProject/README.md](GraduationProject/README.md) for installation
and desktop usage. It requires a graphical environment and is separate from
the ContinualSteg training pipeline.
