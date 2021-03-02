from setuptools import find_packages, setup

setup(name='bomberman_rl',
      version='0.1.0.dev',
      author="Alicia Fortes Machado, Aloysio Galvão Lopes, Iago Martinelli Lopes, Igor Albuquerque Silva",
      description="DQN to play bomberman",
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      keywords="DQN deep learning gym openai reinforcement learning pytorch",
      package_dir={"": "src"},
      packages=find_packages("src"),
      python_requires='>=3.6',
      install_requires=[
            'gym',
            'nptyping'
      ]
)
