test_string = "hello! こんにちは!"

utf8_encoded  = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")

# Printing the encoded raws bytes
print("UTF-8 : ", utf8_encoded, "\n list(UTF-8)", list(utf8_encoded))
print("UTF-16: ", utf16_encoded, "\n list(UTF-16)", list(utf16_encoded))
print("UTF-32: ", utf32_encoded, "\n list(UTF-32)", list(utf32_encoded))

# Looking for the sentences size in each encode, the UTF-8 has the lowest
# numbers of codes, so it should be more efficient for a training

def decode_utf_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# Example that yields incorrect results
# print("Example with wrong results: ", decode_utf_bytes_to_str_wrong(f"hello teste é â".encode("utf-8")))

# The problem with this function shows up when some character requires more than 
# one byte of representation, like "é". The standard UTF-8 format supports 4-bytes 

# print("Non decoding byte sequence: " + b"\xC0\xAF".decode("utf-8"))

# An overlong encoding doesnt work properly, because in UTF-8 you must represent the char with the lowest
# numbers of bytes as possible
