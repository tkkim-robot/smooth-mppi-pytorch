from setuptools import setup

setup(
    name='pytorch_mppi',
    version='0.1.0',
    packages=['pytorch_mppi'],
    url='https://github.com/ktk1501/smooth-mppi-pytorch',
    license='MIT',
    author='Taekyung Kim',
    author_email='ktk1501@kakao.com',
    description='Smooth Model Predictive Path Integral without Smoothing (SMPPI) implemented in pytorch',
    install_requires=[
        'torch',
        'numpy',
        'scipy'
    ],
    tests_require=[
        'gym'
    ]
)
