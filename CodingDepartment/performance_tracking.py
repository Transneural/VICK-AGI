import datetime
import smtplib
from email.mime.text import MIMEText
import psutil
from slack_bolt import App
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import importlib
import os

def track_performance(feedback=False):
    """
    Track the performance of the code generation system.

    Args:
    - feedback (bool): Whether the user provided feedback on the generated code. Default is False.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Write the performance data to a log file
    with open("performance_log.txt", "a") as f:
        if feedback:
            f.write(f"{timestamp}: Generated code was rated as good\n")
        else:
            f.write(f"{timestamp}: Generated code was not rated or was rated as bad\n")

def log_error(error_message):
    """
    Log an error message to a file.

    Args:
    - error_message (str): The error message to log.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Write the error message to a log file
    with open("error_log.txt", "a") as f:
        f.write(f"{timestamp}: {error_message}\n")

def log_warning(warning_message):
    """
    Log a warning message to a file.

    Args:
    - warning_message (str): The warning message to log.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Write the warning message to a log file
    with open("warning_log.txt", "a") as f:
        f.write(f"{timestamp}: {warning_message}\n")

def generate_report():
    """
    Generate a performance report based on the data in the performance log.
    """
    # Get the current date and time
    now = datetime.datetime.now()

    # Define the date range for the report (last 7 days)
    end_date = now.date()
    start_date = end_date - datetime.timedelta(days=7)

    # Open the performance log file and read the contents
    with open("performance_log.txt", "r") as f:
        log_data = f.readlines()

    # Filter the log data to only include entries from the past 7 days
    recent_log_data = [line for line in log_data if start_date <= datetime.datetime.strptime(line.split(":")[0], "%Y-%m-%d %H:%M:%S").date() <= end_date]

    # Calculate the total number of generated code samples
    total_samples = len(log_data)

    # Calculate the number of good and bad samples
    good_samples = len([line for line in recent_log_data if "good" in line])
    bad_samples = len([line for line in recent_log_data if "bad" in line])

    # Calculate the percentage of good and bad samples
    if total_samples > 0:
        good_percentage = round((good_samples / total_samples) * 100, 2)
        bad_percentage = round((bad_samples / total_samples) * 100, 2)
    else:
        good_percentage = 0
        bad_percentage = 0

    # Print the performance report
    print(f"Performance Report ({start_date} - {end_date}):")
    print(f"Total samples: {total_samples}")
    print(f"Good samples: {good_samples} ({good_percentage}%)")
    print(f"Bad samples: {bad_samples} ({bad_percentage}%)")

def send_email(subject, message, recipient):
    """
    Send an email with a specified subject and message.

    Args:
    - subject (str): The subject of the email.
    - message (str): The message to include in the email.
    - recipient (str): The email address of the recipient.
    """
    # Set up the email message
    email = MIMEText(message)
    email['Subject'] = subject
    email['From'] = 'your_email_address@example.com'
    email['To'] = recipient

    # Set up the SMTP server
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.ehlo()
    smtp_server.starttls()
    smtp_server.ehlo()
    smtp_server.login('your_email_address@example.com', 'your_email_password')

    # Send the email
    smtp_server.sendmail('your_email_address@example.com', recipient, email.as_string())
    smtp_server.quit()

def send_slack_message(channel, message):
    """
    Send a message to a specified Slack channel.

    Args:
    - channel (str): The name of the Slack channel.
    - message (str): The message to send.
    """
    # Create a new WebClient instance with the Slack bot token
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

    try:
        # Call the chat.postMessage method using the WebClient
        response = client.chat_postMessage(channel=channel, text=message)

        # Print the response for debugging purposes
        print(response)

    except SlackApiError as e:
        print("Error sending message: {}".format(e))


def get_system_metrics():
    """
    Get system metrics such as CPU usage and memory usage.

    Returns:
    - metrics (dict): A dictionary containing the system metrics.
    """
    # Get CPU usage as a percentage
    cpu_usage = psutil.cpu_percent()

    # Get memory usage as a percentage
    memory_usage = psutil.virtual_memory().percent

    # Construct a dictionary of metrics and return it
    metrics = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage
    }
    return metrics

def check_dependencies(dependencies):
    """
    Check that all required dependencies are installed.

    Args:
    - dependencies (list): A list of required dependencies.

    Returns:
    - missing_dependencies (list): A list of dependencies that are not installed.
    """
    missing_dependencies = []
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            missing_dependencies.append(dependency)
    return missing_dependencies

def check_palindrome(word):
    """
    Checks if a given word is a palindrome (reads the same backwards as forwards)
    :param word: A string to check if it is a palindrome
    :return: True if the word is a palindrome, False otherwise
    """
    return word == word[::-1]

def get_palindromes_from_list(words_list):
    """
    Returns a list of palindromes from a given list of words
    :param words_list: A list of words to check for palindromes
    :return: A list of palindromes from the given list of words
    """
    return [word for word in words_list if check_palindrome(word)]

def main():
    # example usage
    words_list = ['racecar', 'hello', 'madam', 'kayak', 'world']
    palindromes = get_palindromes_from_list(words_list)
    print(palindromes)

def remove_palindromes_from_list(words_list):
    """
    Removes all palindromes from a given list of words
    :param words_list: A list of words to remove palindromes from
    :return: A new list with all palindromes removed
    """
    return [word for word in words_list if not check_palindrome(word)]

if __name__ == '__main__':
    main()

