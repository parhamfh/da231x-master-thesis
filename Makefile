TRACK:=mytrack

run:
	pipenv run python -m sonata audio/$(TRACK).wav

example:
	pipenv run python example.py

test:
	pipenv run python -m pytest -v tests

list-wavs:
	ls -tl output/*/*.wav
