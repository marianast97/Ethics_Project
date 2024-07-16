## HealthyMom App

You can check the web application [here](https://healthy-mom.streamlit.app/).

You can also run the application on your local machine by following these steps:

#### 1. Download Repository

Download the repository to your machine.

```sh
git clone https://github.com/marianast97/Ethics_Project.git
```

#### 2. Setup the Environment (prefer python **3.9.19**)

```sh
conda create -n newenv python=3.9
```

#### 3. Install necessary dependencies

```sh
conda activate newenv
```

- If you are using conda

```sh
conda env update -n newenv -f env.yml
``` 

- OR if you are using pip

```sh
pip install -r requirements.txt
``` 

#### 4. Run the App

```sh
streamlit run Home.py
```

This should open the app in your browser for further exploration

---

#### Data Source
https://archive.ics.uci.edu/dataset/863/maternal+health+risk
