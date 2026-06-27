# Assignment 1 — Chapter 2: BPE encoder
#
# 用法 / Usage:
#   python sol_bpe.py 2.1a        # 运行单个小题
#   python sol_bpe.py 2.2         # 运行某节下所有小题
#   python sol_bpe.py all         # 运行全部
#   python sol_bpe.py             # 列出所有题号

import argparse


# ---------------------------------------------------------------------------
# 2.1 Unicode Standard
# ---------------------------------------------------------------------------

def p_2_1a() -> None:
    """打印 chr(0) —— 空字符 (null)。"""
    print("chr(0):", repr(chr(0)))  # special char "null"


def p_2_1b() -> None:
    """chr(0) 的 __repr__ 与 __str__。"""
    print("repr:", repr(chr(0)))  # '\x00'
    print("str :", str(chr(0)))   # 不可见


def p_2_1c() -> None:
    """字符串拼接中 null 字符的 __repr__ / __str__ 表现。"""
    s = "this is a test" + chr(0) + "string"
    print("repr:", repr(s))  # 'this is a test\x00string'
    print("str :", s)        # this is a test<null>string


# ---------------------------------------------------------------------------
# 2.2 Unicode Encodings
# ---------------------------------------------------------------------------

def UTF8_encode(u: int) -> str:
    if 0 <= u <= 0x7F:
        return chr(u)
    elif 0x80 <= u <= 0x7FF:
        return chr(0xC0 | (u >> 6)) + chr(0x80 | (u & 0x3F))
    elif 0x800 <= u <= 0xFFFF:
        return chr(0xE0 | (u >> 12)) + chr(0x80 | ((u >> 6) & 0x3F)) + chr(0x80 | (u & 0x3F))
    elif 0x10000 <= u <= 0x10FFFF:
        return (chr(0xF0 | (u >> 18)) + chr(0x80 | ((u >> 12) & 0x3F))
                + chr(0x80 | ((u >> 6) & 0x3F)) + chr(0x80 | (u & 0x3F)))
    else:
        raise ValueError("Invalid Unicode value")


def p_2_2a() -> None:
    """为什么训练用 UTF-8 而不是 UTF-16/32：UTF-8 字节序列更短、词表更紧凑。"""
    print(
        "UTF-8 is variable-length: most common characters take 1-2 bytes, "
        "giving a smaller, denser byte vocabulary than UTF-16/32."
    )
    for ch in "a你𝕏":
        print(f"  {ch!r} -> {list(ch.encode('utf-8'))}")


def p_2_2b() -> None:
    """为什么不能逐字节解码 UTF-8：变长编码，多字节字符会失败。"""
    bad = "你好".encode("utf-8")
    print("bytes:", list(bad))
    try:
        print("".join(bytes([b]).decode("utf-8") for b in bad))
    except UnicodeDecodeError as e:
        print("decoding byte-by-byte fails:", e)


def p_2_2c() -> None:
    """一个无法解码为合法 Unicode 的两字节序列示例。"""
    seq = b"\xc0\x00"  # 0xC0 起始但后续不是 continuation byte
    print("bytes:", list(seq))
    try:
        seq.decode("utf-8")
    except UnicodeDecodeError as e:
        print("invalid sequence:", e)


# ---------------------------------------------------------------------------
# 分发 / Dispatch
# ---------------------------------------------------------------------------

SOLUTIONS = {
    "2.1a": p_2_1a,
    "2.1b": p_2_1b,
    "2.1c": p_2_1c,
    "2.2a": p_2_2a,
    "2.2b": p_2_2b,
    "2.2c": p_2_2c,
}


def run(key: str) -> None:
    key = key.lower().replace("(", "").replace(")", "").replace(" ", "")
    if key == "all":
        targets = list(SOLUTIONS)
    elif key in SOLUTIONS:
        targets = [key]
    else:  # 按节前缀匹配，如 "2.2"
        targets = [k for k in SOLUTIONS if k.startswith(key)]
    if not targets:
        raise SystemExit(f"未知题号: {key}\n可用题号: {', '.join(SOLUTIONS)}")
    for k in targets:
        print(f"===== Problem {k} =====")
        SOLUTIONS[k]()
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 BPE 章节习题解答")
    parser.add_argument(
        "problem",
        nargs="?",
        help="题号，如 2.1a / 2.2 / all；留空则列出全部题号",
    )
    args = parser.parse_args()
    if not args.problem:
        print("可用题号:", ", ".join(SOLUTIONS), "| 节前缀 (如 2.1) | all")
        return
    run(args.problem)


if __name__ == "__main__":
    main()
