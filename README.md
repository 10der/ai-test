# ai-test

python3 -m venv .venv
source .venv/bin/activate

pipreqs . --force --ignore .venv

pip install -r requirements.txt
