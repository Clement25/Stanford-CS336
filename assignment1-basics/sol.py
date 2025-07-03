# 2: BPE encoder

# 2.1 Problem 1
#(a)
# print(chr(0))   # special char "null"

#(b)
# print(repr(chr(0))) # \x00

#(c)
# call __repr__
# "this is a test" + chr(0) + "string" -> "this is a test\x00string" 

# call __str__
# print("this is a test" + chr(0) + "string") -> this is a test

# 2.2
# (a)
# Because most characters are represented by 2 bytes, UTF-8 is more efficient to allocate the memory for different unicode characters

def UTF8_encode(u):
    if 0 <= u <= 0x7F:
        return chr(u)
    elif 0x80 <= u <= 0x7FF:
        return chr(0xC0 | (u >> 6)) + chr(0x80 | (u & 0x3F))
    elif 0x800 <= u <= 0xFFFF:
        return chr(0xE0 | (u >> 12)) + chr(0x80 | ((u >> 6) & 0x3F)) + chr(0x80 | (u & 0x3F))
    elif 0x10000 <= u <= 0x10FFFF:
        return chr(0xF0 | (u >> 18)) + chr(0x80 | ((u >> 12) & 0x3F)) + chr(0x80 | ((u >> 6) & 0x3F)) + chr(0x80 | (u & 0x3F))
    else:
        raise ValueError("Invalid Unicode value")

# (b)
# UTF-8 is a variable-length encoding scheme, so we cannot decode the bytestring byte by byte
# def decode_utf8(bytestring: bytes):
#     print(list(bytestring))
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# decode_utf8("你好".encode("utf-8"))

# (c)
# bytestring1 = b"\x00"
# bytestring2 = b"\x01"
# print(bytestring2.decode())

