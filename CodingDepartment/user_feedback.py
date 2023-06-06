import random

def get_feedback_rating():
    """Asks the user to rate their experience and returns the rating."""
    rating = input("How would you rate your experience on a scale of 1-5? ")
    while not rating.isdigit() or int(rating) not in range(1, 6):
        rating = input("Please enter a number between 1 and 5. ")
    return int(rating)

def get_feedback_comments():
    """Asks the user for any comments about their experience and returns the comments."""
    return input("Do you have any comments or suggestions? ")

def get_feedback_name():
    """Asks the user for their name and returns it."""
    return input("What is your name? ")

def get_feedback_email():
    """Asks the user for their email and returns it."""
    email = input("What is your email address? ")
    while "@" not in email:
        email = input("Please enter a valid email address. ")
    return email

def send_feedback_email(email):
    """Sends an email with the user's feedback to the support team."""
    print(f"Sending feedback email to support team at support@domain.com: {email}")

def post_feedback_to_social_media(rating, comments):
    """Posts the user's feedback to a social media platform."""
    print(f"Posting feedback to social media: {rating}/5 stars. {comments}")

def log_feedback_to_database(name, email, rating, comments):
    """Logs the user's feedback to a database."""
    print(f"Logging feedback to database: {name} ({email}) - {rating}/5 stars. {comments}")

def display_thank_you_message(name):
    """Displays a thank you message to the user."""
    print(f"Thank you for your feedback, {name}!")

def ask_for_permission_to_contact():
    """Asks the user for permission to contact them in the future."""
    permission = input("Can we contact you in the future for further feedback? (y/n) ")
    return permission.lower() == "y"

def ask_for_feedback_channel():
    """Asks the user for their preferred feedback channel."""
    channels = ["email", "social media", "phone"]
    channel = ""
    while channel not in channels:
        channel = input(f"What is your preferred feedback channel? ({', '.join(channels)}) ")
    return channel

def get_feedback():
    """Runs the complete feedback process and returns the user's feedback."""
    name = get_feedback_name()
    rating = get_feedback_rating()
    comments = get_feedback_comments()
    email = get_feedback_email()
    send_feedback_email(email)
    post_feedback_to_social_media(rating, comments)
    log_feedback_to_database(name, email, rating, comments)
    display_thank_you_message(name)
    permission_to_contact = ask_for_permission_to_contact()
    if permission_to_contact:
        feedback_channel = ask_for_feedback_channel()
        if feedback_channel == "email":
            send_feedback_email(email)
        elif feedback_channel == "social media":
            post_feedback_to_social_media(rating, comments)
        elif feedback_channel == "phone":
            phone_number = input("What is your phone number? ")
            print(f"Calling {phone_number} to gather further feedback.")
    return {"name": name, "rating": rating, "comments": comments, "email": email, "permission_to_contact": permission_to_contact}
