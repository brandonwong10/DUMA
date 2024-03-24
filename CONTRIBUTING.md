# CONTRIBUTING

Welcome to StudyGen.ai! We appreciate your interest in contributing to our project. Your contributions play a vital role in making StudyGen.ai successful and valuable for the community.

1. Fork the project's repository and create your branch for changes.
Make the desired changes in your branch.
Ensure that your changes adhere to the project's coding standards and guidelines.
If required, sign the Contributor License Agreement (CLA) and note the project's Code of Conduct.
Submit a pull request, indicating that you have a CLA on file and detailing the changes you've made.
If you have a different process for small fixes, please let us know in your pull request description.

## Installation

To get started with Code Racer locally, follow these steps

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
   pip install langchain_openai
   pip install langchain_text_splitters
   pip install streamlit
   pip install python-dotenv
   pip install PyPDF2
   pip install langchain
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

## Pulling in changes from upstream
