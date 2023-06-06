import subprocess
import spacy
import requests
import json
import re
import execjs

available_languages = ["Python", "Java", "C++", "JavaScript"]

def select_language():
    """
    Prompts the user to enter some code and automatically detects the programming language.

    Returns:
        str: The detected programming language.
    """
    # Load the pre-trained language detection model
    nlp = spacy.load('en_core_web_sm')

    # Prompt the user to enter some code
    code = input("Enter some code: ")

    # Use the language detection model to predict the language
    doc = nlp(code)
    detected_language = doc._.languages[0].capitalize()

    # Check if the detected language is in the list of available languages
    if detected_language in available_languages:
        return detected_language
    else:
        print(f"Detected language ({detected_language}) is not in the list of available languages.")
        return None


def generate_code(language):
    """
    Generates a code snippet for the given programming language.

    Args:
        language (str): The programming language to generate code for.

    Returns:
        str: The generated code snippet.
    """
    if language == "Python":
        return "print('Hello, world!')"
    elif language == "Java":
        return "System.out.println('Hello, world!');"
    elif language == "C++":
        return "#include <iostream>\nint main() { std::cout << 'Hello, world!' << std::endl; return 0; }"
    elif language == "JavaScript":
        return "console.log('Hello, world!');"
    else:
        return None


def execute_code(code, language):
    """
    Executes the given code snippet for the specified programming language.

    Args:
        code (str): The code snippet to execute.
        language (str): The programming language of the code snippet.

    Returns:
        str: The output of the executed code.
    """
    if language == "Python":
        try:
            exec(code)
        except Exception as e:
            return str(e)
    elif language == "Java":
        try:
            # Compile the Java code
            compile_command = f"javac code.java"
            subprocess.check_output(compile_command, shell=True)

            # Execute the Java code
            execute_command = f"java code"
            output = subprocess.check_output(execute_command, shell=True)

            return output.decode()
        except subprocess.CalledProcessError as e:
            return e.output.decode()
    elif language == "C++":
        try:
            # Compile the C++ code
            compile_command = f"g++ -o code code.cpp"
            subprocess.check_output(compile_command, shell=True)

            # Execute the C++ code
            execute_command = f"./code"
            output = subprocess.check_output(execute_command, shell=True)

            return output.decode()
        except subprocess.CalledProcessError as e:
            return e.output.decode()
    elif language == "JavaScript":
        try:
            execjs.eval(code)
        except Exception as e:
            return str(e)
    else:
        return None


def save_code(code, filename):
    """
    Saves the given code snippet to a file with the specified filename.

    Args:
        code (str): The code snippet to save.
        filename (str): The filename to save the code snippet to.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        f.write(code)


def load_code(filename):
    """
    Loads the code snippet from a file with the specified filename.

    Args:
        filename (str): The filename to load the code snippet from.

    Returns:
        str: The loaded code.
    """
    with open(filename, 'r') as f:
        code = f.read()
    return code


