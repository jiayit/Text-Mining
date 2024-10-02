import spacy

def similarity():

    # Load the en_core_web_md pipeline
    nlp = spacy.load("en_core_web_md")

    # Process a text
    doc = nlp("Two bananas in pyjamas")

    # Get the vector for the token "bananas"
    bananas_vector = doc[1].vector
    print(bananas_vector)

if __name__ == '__main__':
    similarity()