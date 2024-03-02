import setuptools

with open("README.md", "r", encoding= 'utf-8') as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "data_science/tree/main/Kidney%20Detection"
Author_user_name = "zacharias1219"
src_repo = "cnnClassifier"
Author_email = "rickoshade1891@gmail.com"

setuptools.setup(
    name= REPO_NAME,
    version=__version__,
    author=Author_user_name,
    author_email=Author_email,
    description="A small package for kidney detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{Author_user_name}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{Author_user_name}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)