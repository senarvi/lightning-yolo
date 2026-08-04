# Agent Instructions

## Software Design Principles

Optimize for the quality of the final software product, not for the size of the diff. A change is done when the
codebase as a whole has a clean architecture, elegant code, and a coherent design — not when the requested behavior
merely works.

- **Prefer the best design over minimal edits.** Do not bolt on new behavior as an isolated block wedged into an
  existing function or file just because it is the smallest possible change. Integrate it where it belongs, even if
  that means editing several files or reshaping existing code.
- **Do not preserve backward compatibility for its own sake.** This is an evolving codebase, not a published library.
  Remove or change interfaces, arguments, and defaults freely when a better design calls for it. Do not keep dead
  parameters, compatibility shims, or aliases "just in case".
- **Refactor as part of the change.** After adding or removing something, consider whether the surrounding code should
  be refactored so the result still reads as if it had been designed that way from the start. Update call sites, tests,
  and types to match.
- **Maintain consistency across equivalent abstractions.** When a change improves an interface, ownership boundary, or
  naming pattern in one part of the project, carry that consistency through the corresponding code paths unless there is
  a clear reason not to. Avoid leaving parallel implementations with needlessly different structure.
- **Minimize interfaces and maximize cohesion.** Keep public surfaces small: expose only what callers need, and keep
  related logic together. A function or class should have a single, clear responsibility.
- **Avoid single-use micro-helpers.** Do not extract tiny helper functions that are called from only one or two places
  and exist mainly to split work into pieces. A helper earns its place by being genuinely reusable across the codebase
  or by encapsulating a meaningful, well-named concept. Prefer inlining trivial logic, or designing a general reusable
  function, over a proliferation of narrow one-off helpers.

## Docstrings

All functions, methods, and classes must have a docstring, including private helpers. Document all arguments and return
values using the Google-style format already present in the codebase:

```python
def sum(x: int, y: str) -> bool:
    """Calculates the sum of two variables.

    Args:
        x: Description of x.
        y: Description of y.

    Returns:
        Description of the return value.

    """
```

For classes, document constructor arguments in the class docstring. For methods, document method arguments and return
values where applicable.

Docstrings use reStructuredText. Format numeric tensor shapes using double backticks (literal code); use single
backticks for named quantities and format conventions (emphasis).

A shape is a concrete list or tuple of dimension sizes. Examples:

```
``[batch_size, num_channels, height, width]``
``[batch_size, N, max_targets]``
```

A format name, coordinate label, math expression, or dimension-order convention is not a shape — it names what the
values mean. Examples:

```
`(width, height)` ← dimension-order convention
`N × N`           ← math expression
```

## Unit Test Conventions

Prefer one concise test function per public function or method under test. The test should cover the basic behavior of
that subject without trying to encode every scenario in the function name.

Do not add tests for incidental validation details, such as a class rejecting an invalid argument, unless that behavior
is part of an important public contract or has caused a real bug.

Use `pytest.mark.parametrize` when the same behavior should be checked across multiple inputs, modes, or label forms.
This is preferred over writing separate test functions for each small variation.

Split into additional test functions only when the cases become too large for one readable test, or when a particular
argument or mode needs its own setup and assertions. In that case, keep the core behavior in the primary test and name
the extra test after the subject plus the argument being varied — for example, `test_tal_matching` covers the core
behavior while `test_tal_matching_input_is_normalized` covers the `input_is_normalized` argument with parametrized
cases.

Derive the test name from the subject under test, not from a prose description of the scenario:

- Free function `foo` → `test_foo`
- Method `Bar.baz` → `test_bar_baz`

For classes whose primary behavior is exposed through `__call__`, use the class name as the subject
(`test_highest_iou_matching`, `test_sim_ota_matching`, `test_tal_matching`).

## Type Annotations

- Annotate all production functions, methods, class attributes, and non-obvious local variables. Prefer precise types
  that communicate the contract rather than broad `object` or `Any`.
- Use built-in generic syntax and union operators: `list[Tensor]`, `dict[str, float]`, `tuple[int, int]`, and
  `Tensor | None`. Do not import legacy aliases such as `List`, `Dict`, `Tuple`, or `Optional` from `typing`.
- Import abstract input collection types from `collections.abc`: accept `Sequence[T]`, `Mapping[K, V]`, `Iterable[T]`,
  or `Callable[...]` when mutation or a concrete container is not required. Return concrete types when callers depend
  on a concrete result.
- Assume a recent Python version where forward references and self-references work without workarounds. Write
  annotations naturally instead of adding `from __future__ import annotations` or quoting type annotations.
- Declare meaningful aliases with the `type` statement, such as `type PredictionDict = dict[str, Tensor]`. Use an
  alias for a domain concept or a repeated, non-trivial type; do not introduce aliases that merely abbreviate a simple
  built-in type.
- Model absence with `T | None`, finite alternatives with `Literal[...]`, and a method returning the same subclass
  with `Self`. Use `Never` only for code paths that cannot return normally.
- Treat `Any` as an escape hatch, not a default. Prefer `object` for an intentionally opaque value, a protocol or
  union for a constrained interface, and a generic type variable for a value whose type must be preserved. Narrow
  union types with `isinstance`, explicit `None` checks, or `TypeGuard` rather than using casts to silence errors.
- Use `cast` only when runtime behavior already proves a fact the type checker cannot infer. Keep casts next to that
  proof and avoid using them to mask an incorrect or underspecified interface.
- Keep `# type: ignore[...]` as a last resort. It must name the specific mypy error code and include a short reason;
  fix the type boundary, missing stub, or annotation instead whenever practical. Mypy reports unused ignores.
- Use `@overload` only when a function's return type genuinely depends on distinct input forms. Keep the concrete
  implementation immediately below the overloads and annotate it with their shared implementation type.
