# SAM-Label

This is an image labeling tool based on Segment-Anything.

![Click The Image](assets/SAM-label/click.png)

![Labels](assets/SAM-label/labels.png)

## Installation

### 1. Setup the environment for Segment-Anything

Follow the installation instruction in [README](README-Segment-Anything.md) and setup the environment.

### 2. Install PyQt and solve the environment conflict

```bash
pip install PyQt5 matplotlib
pip uninstall opencv-python
pip install opencv-python-headless
```

## GettingStarted

### 1. Download Checkpoint

Follow the installation instruction in [README](README-Segment-Anything.md) and download the checkpoint.

### 2. Start the Program

```bash
python ./gui.py
```

Left mouse to add positive point (yellow point in image), Right mouse to add negative point (blue point in image).

![get startted](assets/SAM-label/run.png)

