from pathlib import Path
import numpy as np
import dvm

@dvm.kernel
def upstream_add(k, x, y):
    a = k.load(x, dvm.float32)
    b = k.load(y, dvm.float32)
    c = k.add(a, b)
    return k.store(c)

def main() -> None:
    x = np.full([32, 32], 0.1, np.float32)
    y = np.full([32, 32], 0.3, np.float32)
    upstream_add.codegen()
    text = upstream_add.das()
    out = Path("tests/fixtures/upstream_add_trace.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    out.write_text(text)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
