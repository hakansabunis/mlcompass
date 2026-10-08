"""Every number in the 12-page version must already appear in the extended
version (../tse_latex/main.tex), which the scripts generate. Prints the numbers
of the short source that the long source does not contain."""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
short = (HERE / "main.tex").read_text(encoding="utf-8")
long_ = (HERE.parent / "tse_latex" / "main.tex").read_text(encoding="utf-8")


def strip(tex: str) -> str:
    tex = tex[tex.index(r"\begin{document}"):]
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\(cite[a-z]*|ref|label|input|includegraphics|bibliography\w*)(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = tex.replace("{,}", ",").replace("\\,", " ").replace("$", " ")
    return tex


def numbers(tex: str) -> list[str]:
    # integers with thousands separators, decimals, and 10^{n} powers
    out = re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?", tex)
    return out


long_nums = set(numbers(strip(long_)))
missing = sorted({n for n in numbers(strip(short)) if n not in long_nums},
                 key=lambda x: float(x.replace(",", "")))
print(f"{len(set(numbers(strip(short))))} distinct numbers in the short version; "
      f"not in the extended version: {missing or 'none'}")
for n in missing:
    for m in re.finditer(re.escape(n), strip(short)):
        ctx = re.sub(r"\s+", " ", strip(short)[max(0, m.start() - 70):m.end() + 50])
        print(f"  {n}: ...{ctx}...")
        break


def cited(tex: str) -> set[str]:
    return {k.strip() for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", tex) for k in m.group(1).split(",")}


s_keys, l_keys = cited(short), cited(long_)
print(f"references: {len(s_keys)} cited here, {len(l_keys)} in the extended version")
print("  only here:", sorted(s_keys - l_keys))
print("  extended version only:", sorted(l_keys - s_keys))
