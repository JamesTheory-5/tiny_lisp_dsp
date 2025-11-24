# tiny_lisp_dsp
```python
# tiny_lisp_dsp.py
#
# Tiny Lisp-like DSL that compiles to Numba-jitted DSP tick functions
# using tuple state and (x, state, params) -> (y, new_state).
#
# The DSL is intentionally tiny and opinionated:
#
#   (defdsp name
#     (state s1 s2 ...)
#     (params p1 p2 ...)
#     (tick
#       EXPR))
#
# Where EXPR evaluates to a tuple:
#   (output, new_s1, new_s2, ...)
#
# Supported forms in EXPR:
#   - numbers: 0, 1.0, etc
#   - symbols: input, and any state/param/let variables
#   - (+ e1 e2 ...), (- e1 e2 ...), (* e1 e2 ...), (/ e1 e2)
#   - (let ((v1 e1) (v2 e2) ...) BODY)
#   - (tuple e1 e2 ...)
#
# Example:
#
#   (defdsp onepole
#     (state s)
#     (params a b)
#     (tick
#       (let ((y (+ (* a input) (* b s))))
#         (tuple y y))))
#
# Compiles to:
#
#   @njit(cache=False, fastmath=True)
#   def onepole_tick(x, state, params):
#       s = state[0]
#       a = params[0]
#       b = params[1]
#       y = (a * x + b * s)
#       res = (y, y)
#       out = res[0]
#       new_state = res[1:]
#       return out, new_state
#
# Then you can run blocks by looping over onepole_tick.


import re
import textwrap
import numpy as np
from numba import njit


# ============================================================
# 1. Tokenizer / Parser for S-expressions
# ============================================================

def _tokenize(src: str):
    # Add spaces around parens, then split
    spaced = src.replace("(", " ( ").replace(")", " ) ")
    tokens = spaced.split()
    return tokens


def _parse_tokens(tokens):
    """Recursive descent S-expression parser -> nested Python lists."""
    if len(tokens) == 0:
        raise SyntaxError("Unexpected EOF while reading")

    token = tokens.pop(0)

    if token == "(":
        lst = []
        while len(tokens) > 0 and tokens[0] != ")":
            lst.append(_parse_tokens(tokens))
        if len(tokens) == 0:
            raise SyntaxError("Missing ')'")
        tokens.pop(0)  # remove ')'
        return lst
    elif token == ")":
        raise SyntaxError("Unexpected ')'")
    else:
        # atom
        return _atom(token)


def _atom(token):
    # int?
    try:
        return int(token)
    except ValueError:
        pass
    # float?
    try:
        return float(token)
    except ValueError:
        pass
    # symbol
    return token


def parse_sexpr(src: str):
    tokens = _tokenize(src)
    exprs = []
    while tokens:
        exprs.append(_parse_tokens(tokens))
    if len(exprs) == 1:
        return exprs[0]
    return exprs


# ============================================================
# 2. Lisp → Python code generator (restricted DSL)
# ============================================================

class CodegenError(Exception):
    pass


def _compile_expr(expr, ctx, lines, gensym):
    """
    Compile a Lisp expression into a Python expression string.
    - ctx: dict symbol -> Python name
    - lines: list of emitted 'stmt' strings (for let)
    - gensym: list with single int counter for unique tmp names
    """
    # numbers
    if isinstance(expr, (int, float)):
        return repr(expr)

    # symbol
    if isinstance(expr, str):
        if expr == "input":
            return "x"
        if expr in ctx:
            return ctx[expr]
        # fallback: raw symbol as Python name
        return expr

    # list form
    if not isinstance(expr, list) or len(expr) == 0:
        raise CodegenError(f"Bad expression: {expr!r}")

    head = expr[0]

    # ---------------------------
    # let
    # (let ((v1 e1) (v2 e2) ...) body)
    # ---------------------------
    if head == "let":
        if len(expr) != 3:
            raise CodegenError("(let ...) must be (let ((v val) ...) body)")
        bindings = expr[1]
        body = expr[2]
        if not isinstance(bindings, list):
            raise CodegenError("let bindings must be a list")

        new_ctx = dict(ctx)
        for b in bindings:
            if not isinstance(b, list) or len(b) != 2:
                raise CodegenError("each let binding must be (name expr)")
            name = b[0]
            value_expr = b[1]
            py_rhs = _compile_expr(value_expr, new_ctx, lines, gensym)
            py_name = name  # simple 1:1 mapping
            lines.append(f"{py_name} = {py_rhs}")
            new_ctx[name] = py_name

        return _compile_expr(body, new_ctx, lines, gensym)

    # ---------------------------
    # arithmetic
    # (+ e1 e2 ...), (- e1 e2 ...), (* e1 e2 ...), (/ e1 e2)
    # ---------------------------
    if head in ("+", "-", "*", "/"):
        if len(expr) < 2:
            raise CodegenError(f"({head}) needs at least one arg")
        args = [_compile_expr(a, ctx, lines, gensym) for a in expr[1:]]

        if head == "-" and len(args) == 1:
            return f"(-{args[0]})"

        op = {"+" : "+", "-" : "-", "*" : "*", "/" : "/"}[head]
        return "(" + f" {op} ".join(args) + ")"

    # ---------------------------
    # tuple: (tuple e1 e2 ...)
    # ---------------------------
    if head == "tuple":
        parts = [_compile_expr(a, ctx, lines, gensym) for a in expr[1:]]
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            # single-element tuple syntax
            return f"({parts[0]},)"
        return "(" + ", ".join(parts) + ")"

    raise CodegenError(f"Unknown form: {head}")


def compile_dsp(dsl_source: str, *, prefix_njit=True):
    """
    Compile a single (defdsp ...) form into a Numba-jitted tick function.

    DSL:
        (defdsp name
          (state s1 s2 ...)
          (params p1 p2 ...)
          (tick EXPR))

    The EXPR must evaluate to a tuple:
        (output, new_s1, new_s2, ...)

    Returns:
        tick_fn: Python function decorated with @njit(cache=False, fastmath=True)
    """
    ast = parse_sexpr(dsl_source)

    if not isinstance(ast, list) or len(ast) < 4:
        raise CodegenError("Expected (defdsp name (state ...) (params ...) (tick ...))")

    if ast[0] != "defdsp":
        raise CodegenError("Top-level form must be (defdsp ...)")

    name = ast[1]
    state_form = ast[2]
    params_form = ast[3]
    tick_form = ast[4] if len(ast) > 4 else None

    if not isinstance(name, str):
        raise CodegenError("defdsp name must be a symbol")

    # (state s1 s2 ...)
    if not isinstance(state_form, list) or len(state_form) < 1 or state_form[0] != "state":
        raise CodegenError("Expected (state s1 s2 ...)")

    state_names = state_form[1:]

    # (params p1 p2 ...)
    if not isinstance(params_form, list) or len(params_form) < 1 or params_form[0] != "params":
        raise CodegenError("Expected (params p1 p2 ...)")
    param_names = params_form[1:]

    # (tick EXPR)
    if not isinstance(tick_form, list) or len(tick_form) < 2 or tick_form[0] != "tick":
        raise CodegenError("Expected (tick EXPR)")
    tick_expr = tick_form[1]

    tick_name = f"{name}_tick"

    # Build Python source for tick function
    lines = []
    indent = " " * 4
    body_lines = []

    ctx = {}
    gensym = [0]

    # Unpack state
    for idx, s in enumerate(state_names):
        body_lines.append(f"{s} = state[{idx}]")
        ctx[s] = s

    # Unpack params
    for idx, p in enumerate(param_names):
        body_lines.append(f"{p} = params[{idx}]")
        ctx[p] = p

    # Compile the body expression
    expr_str = _compile_expr(tick_expr, ctx, body_lines, gensym)

    # Evaluate body into res; then split into output and new_state tuple
    body_lines.append(f"res = {expr_str}")
    body_lines.append("out = res[0]")
    body_lines.append("new_state = res[1:]")
    body_lines.append("return out, new_state")

    # Put it together
    fn_header = f"def {tick_name}(x, state, params):"
    full_body = "\n".join(indent + ln for ln in body_lines)
    fn_src = fn_header + "\n" + full_body + "\n"

    if prefix_njit:
        decorated_src = (
            "@njit(cache=False, fastmath=True)\n" + fn_src
        )
    else:
        decorated_src = fn_src

    # Exec into a dedicated namespace
    ns = {
        "njit": njit,
        "np": np,
    }
    exec(decorated_src, ns)

    tick_fn = ns[tick_name]
    # For debugging, you might want to see the generated code:
    # print("Generated code:\n", decorated_src)
    return tick_fn


# ============================================================
# 3. Simple block processor (Python loop calling jitted tick)
# ============================================================

def process_block(x, state, params, tick_fn):
    """
    Run a block through a compiled tick function.

    x      : 1D ndarray of input
    state  : tuple of initial state
    params : tuple of params
    tick_fn: (x_sample, state, params) -> (y_sample, new_state)
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    y = np.empty_like(x)
    s = state
    for i in range(N):
        yi, s = tick_fn(x[i], s, params)
        y[i] = yi
    return y, s


# ============================================================
# 4. Smoke test: one-pole lowpass filter + plots
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Define a 1-pole lowpass in Lisp DSL.
    #
    # Continuous-time heuristic:
    #   y[n] = a * x[n] + b * y[n-1]
    #
    # We'll store previous output in state s.
    #
    # State: (s)
    # Params: (a b)
    # Tick returns (y, new_s)
    # --------------------------------------------------------
    onepole_dsl = r"""
    (defdsp onepole
      (state s)
      (params a b)
      (tick
        (let ((y (+ (* a input) (* b s))))
          (tuple y y)))
    )
    """

    onepole_tick = compile_dsp(onepole_dsl)

    # Filter design: simple exponential smoothing
    fs = 48000.0
    cutoff = 1000.0
    # crude mapping: a = 1 - exp(-2π f0 / fs), b = 1 - a
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff / fs)
    a = alpha
    b = 1.0 - alpha

    # Impulse for frequency response
    N = 16384
    x = np.zeros(N, dtype=np.float64)
    x[0] = 1.0

    init_state = (0.0,)
    params = (a, b)

    y, final_state = process_block(x, init_state, params, onepole_tick)

    # --------------------------------------------------------
    # Plot impulse response
    # --------------------------------------------------------
    t = np.arange(N) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t[:1000], y[:1000])
    plt.title("One-pole Lowpass (Lisp → Numba) – Impulse Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # --------------------------------------------------------
    # Plot magnitude frequency response
    # --------------------------------------------------------
    # FFT
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag_db = 20.0 * np.log10(np.maximum(1e-12, np.abs(Y)))

    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, mag_db)
    plt.title("One-pole Lowpass (Lisp → Numba) – Magnitude Response")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True, which="both", ls=":")

    plt.tight_layout()
    plt.show()

    # Optional: listen to impulse response scaled down a bit
    try:
        import sounddevice as sd

        sd.play(y * 0.2, int(fs))
        sd.wait()
    except Exception as e:
        print("Audio playback unavailable:", e)

```
