pip uninstall -y dist\HyTime-0.1-py3-none-any.whl
if exist setup.py ( python setup.py sdist bdist_wheel )
pip install dist\HyTime-0.1-py3-none-any.whl