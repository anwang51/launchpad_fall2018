SPACE = 32
EXCLAMATION = 33
PERIOD = 46
COMMA = 44
APOSTROPHE = 39
QUESTION = 63
SEMI_COLON = 59
NEW_LINE = 10
COLON = 58
DASH = 45
AMPERSAND = 38
OPEN_BRACKET = 91
CLOSE_BRACKET = 93

PUNCTUATION = set()
PUNCTUATION.add(SPACE)
PUNCTUATION.add(EXCLAMATION)
PUNCTUATION.add(PERIOD)
PUNCTUATION.add(COMMA)
PUNCTUATION.add(APOSTROPHE)
PUNCTUATION.add(QUESTION)
PUNCTUATION.add(SEMI_COLON)
PUNCTUATION.add(NEW_LINE)
PUNCTUATION.add(COLON)
PUNCTUATION.add(DASH)
PUNCTUATION.add(AMPERSAND)
PUNCTUATION.add(OPEN_BRACKET)
PUNCTUATION.add(CLOSE_BRACKET)

def is_legal_character(c):
	v = ord(c)
	if v >= 65 and v <= 90: # upper case
		return True
	elif v >= 97 and v <= 122: # lower case
		return True
	elif v >= 48 and v <= 57: # numbers
		return True
	elif v in PUNCTUATION:
		return True
	return False

f = open("shakespeare.txt").read()
r = open("processed_shakespeare.txt", "w")
for c in f:
	if is_legal_character(c):
		r.write(c)
r.flush()