# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import spacy
from spacy.matcher import Matcher



def textMining():
    nlp = spacy.blank("en")
    doc = nlp("This is a sentence.")

    # Print the document text
    print(doc.text)

def token():
    nlp = spacy.blank("en")
    doc = nlp("I like tree kangaroos and narwhals.")
    first_token = doc[0]
    print(first_token.text)

def sliceToken():
    nlp = spacy.blank("en")
    doc = nlp("I like tree kangaroos and narwhals.")
    tree_kangaroos = doc[2:4]
    print(tree_kangaroos.text)

def lexicalAttributes():
    nlp = spacy.blank("en")
    doc = nlp("In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are.")
    for token in doc:
        if token.like_num:
            next_token = doc[token.i + 1]
            if next_token.text == "%":
                print("Percentage found:", token.text)

def loadPipiline():
    npl = spacy.load("en_core_web_sm")
    text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
    doc = npl(text)
    print(doc.text)

def matcher():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Upcoming iPhone X release date leaked as Apple reveals pre-orders")

    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)

    # Create a pattern matching two tokens: "iPhone" and "X"
    pattern = [{"TEXT": "iphone"}, {"TEXT": "X"}]

    # Add the pattern to the matcher
    matcher.add("IPHONE_X_PATTERN", pattern)

    # Use the matcher on the doc
    matches = matcher(doc)
    print("Matches:", [doc[start:end].text for match_id, start, end in matches])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    textMining()
    token()
    sliceToken()
    lexicalAttributes()
    loadPipiline()
    matcher()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
