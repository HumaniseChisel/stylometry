# Stylometry: Keystroke Dynamics Classification

Identify users based on their keystroke dynamics using a deep Recurrent Neural Network (Bi-LSTM).

See our [report](report.pdf) and [Jupyter Notebook](backend/train.ipynb) for more details.

The website for data collection and inference is deployed live at https://stylometry.pages.dev.

## Training Notebook Steps
Start a venv in the backend directory
```
cd ./backend
python -m venv .venv-notebook
source ./.venv-notebook/bin/activate
```
Install the dependencies
```
pip install -r notebook-requirements.txt
```
Open the notebook in your IDE and ensure you use the created venv as the notebook kernel.

<img width="970" height="529" alt="image" src="https://github.com/user-attachments/assets/19f04a8f-211e-4052-be4d-e4e492c79de3" />
<img width="970" height="485" alt="image" src="https://github.com/user-attachments/assets/792665df-eb30-40c2-a3f2-cc240eff4138" />


## Backend
Run the backend with
```
python api/src/api/main.py
```
or
```
uv run --package api api/src/api/main.py
```

## Training Script
This is the same as the notebook.
Run it with
```
python3 -m model.main
```
or
```
uv run -m model.main
```
