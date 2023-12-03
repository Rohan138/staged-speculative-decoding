from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)

setup(
    name="staged-speculative-decoding",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    license="MIT",
    description="Staged Speculative Decoding",
    author="Rohan Potdar",
    author_email="rohanpotdar138@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/Rohan138/staged-speculative-decoding",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformers",
        "speculative decoding",
    ],
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
