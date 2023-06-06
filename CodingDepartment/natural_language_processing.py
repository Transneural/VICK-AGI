import spacy

# Load the large English NLP model
nlp = spacy.load('en_core_web_lg')

def process_input(text):
    """
    Processes the input text using NLP techniques.

    Args:
        text: The input text to be processed.

    Returns:
        A dictionary containing the processed information.
    """
    # Create a Doc object from the input text
    doc = nlp(text)

    # Extract the named entities
    named_entities = []
    for ent in doc.ents:
        named_entities.append((ent.text, ent.label_))

    # Extract the noun chunks
    noun_chunks = []
    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)

    # Return the processed information as a dictionary
    processed_info = {'text': text,
                      'named_entities': named_entities,
                      'noun_chunks': noun_chunks}

    return processed_info
