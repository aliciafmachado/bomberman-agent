from setuptools import find_packages, setup

setup(name='bomberman_rl',
      version='0.1.0.dev',
      author="Alicia Fortes Machado, Aloysio Galvão Lopes, Iago Martinelli Lopes, Igor Albuquerque Silva",
      description="DQN to play bomberman",
      long_description=open("README.md", "r").read(),
      long_description_content_type="text/markdown",
      keywords="DQN deep learning gym openai reinforcement learning pytorch",
      package_dir={"": "src"},
      packages=find_packages("src"),
      python_requires='>=3.6',
      install_requires=[
                'gym>=0.18.0',
                'nptyping>=1.4.0',
                'pygame>=2.0.0',
                'matplotlib>=3.0.0',
                'numpy>=1.19',
                'torch',
                'torchvision',
                'torch',
                'torchvision'
            ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest']
      )
