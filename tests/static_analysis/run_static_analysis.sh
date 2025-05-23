# get dependencies
pip install -r ./requirements.txt

# check cyclomatic complexity
python -m mccabe --min 3 ./../src/**/*.py

# SAST (static application security testing)
bandit -r ./../src

mypy 