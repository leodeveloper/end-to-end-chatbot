from setuptools import find_packages,setup

setup(
    name="chatcreator",
    version="0.0.1",
    author="leodeveloper",
    author_email="leodeveloper@gmail.com",
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2","chromadb"],
    packages=find_packages()
)