# The chr(0) returns a \n line

print(f"chr(0): {chr(0)}")
# The chr function returns a character based on a specific unicode

print(f"__repr__: {chr(0).__repr__()}))")
# The __repr__ differs bringing the code representation as a string

print(f"Testing the char(0), this is a test {chr(0)} string!")
# This chr(0) just add a space in the sentence

for i in range(1, 10000, 500):
    print("-"*50)
    print(f"chr({i}): {chr(i)}")
    print(f"__repr__: {chr(i).__repr__()}")
 
