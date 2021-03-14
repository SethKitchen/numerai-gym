upload:
	make clean
	python setup.py sdist bdist_wheel && twine upload dist/*
clean:
	python setup.py clean --all
	pyclean .
	rm -rf *.pyc __pycache__ build dist gym_numerai.egg-info gym_numerai/__pycache__ gym_numerai/units/__pycache__ tests/__pycache__ tests/reports docs/build
