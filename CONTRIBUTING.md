# CONTRIBUTING

Welcome to StudyGen.ai! We appreciate your interest in contributing to our project. Your contributions play a vital role in making StudyGen.ai successful and valuable for the community.

## Installation

To get started with StudyGen.ai locally, follow these steps

1. Fork the repo

2. Create a branch for changes

3. Clone your fork

   ```sh
    git clone https://github.com/<YOUR_GITHUB_ACCOUNT_NAME>/StudyGen.ai.git
   ```

4. Navigate to the project directory

   ```sh
   cd StudyGen.ai
   ```

5. Create a .env file inside the project's packages/app directory.

6. Copy and paste variables: OPENAI_API_KEY = (Enter your OpenAi key here)

7. Install all required packages

   ```sh
   pip install langchain_openai, langchain_text_splitters, streamlit, python-dotenv, PyPDF2, langchain
   ```

8. Run the program with

   ```sh
   streamlit run pdfanalysis.py
   ```

## Working on New Features

If you want to work on a new feature, follow these steps.

1. Fork the repo
2. Clone your fork
3. Checkout a new branch
4. Do your work
5. Commit
6. Push your branch to your fork
7. Go into github UI and create a PR from your fork & branch, and merge it into upstream MAIN

## Contribution that can be made

1. Proper website for the MVP
2. Adding additional APIS for specific topics that OPENAI is weaker at
