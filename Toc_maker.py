import sys

args = sys.argv

assert len(args) == 2, "引数（.md ファイルのパス）を指定してください"
assert args[1].endswith(".md"), "markdown ファイルを指定してください"

md_path = args[1]
contents = None
with open(md_path, "r", encoding="utf-8_sig") as f:
    contents = [s.strip() for s in f.readlines()]
    print(len(contents))

# contents = ["# hello", "## world", "### trigger"] # test

toc = []
for line in contents:
    if not line.startswith("#"):
        continue
    t = ""
    if line.startswith("# "):
        line = line[2:]  # `# `以降の文字列
        t = f"- [{line}](#{line})"
    elif line.startswith("## "):
        line = line[3:]  # `## `以降の文字列
        t = f"  - [{line}](#{line})"
    elif line.startswith("### "):
        line = line[4:]  # `## `以降の文字列
        t = f"    - [{line}](#{line})"
    toc.append(t)

with open("toc.txt", "w", encoding="utf-8_sig") as f:
    print(*toc, sep="\n", file=f)
