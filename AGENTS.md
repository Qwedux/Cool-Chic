1. The main goal of the implementation is referential transparency and type safety. The main tools to use are:
- frozen dataclasses, optionally with kw_only=True if they need to hold data. These can be used in match case statements instead of ifs
- typing.NewType for type aliases, zero overhead and protected against accidental argument swapping, but cannot be used in match case statements
- don't use literals
- for compatibility with fvcore FlopCountAnalysis you can use TypedDict for the output of the forward pass of the model
- don't use Enums
- Use Sequence[T] instead of list[T]
- Use Mapping[K, V] instead of dict[K, V]

2. Use assert_never at the end of all match case statements, even if they are exhaustive, this protects against new cases being added silently.

3. The environment uses conda, activate it with `conda activate cool_chic_venv`

4. Take pytorch as frozen legacy code, don't change it.

5. Code inside of domain should never raise exceptions or assertions, unless the user explicitly leaves comment `# NOTE: Exception or Assertion allowed here`

6. Use specialized types whenever possible.

Good example:
```python
def compute_area(square_side_length: SquareSideLength) -> Area:
    return square_side_length * square_side_length
```

Bad example:
```python
def compute_area(square_side_length: int) -> int:
    return square_side_length * square_side_length
```

7. Type should reflect validation success. If type T goes into the validation, the result should be T_validated | ValidationError. Optionally it might make sense to return T_invalid instead of ValidationError.

Good example:
```python
def validate_square_side_length(square_side_length: SquareSideLength) -> SquareSideLengthValidated | ValidationError:
    if square_side_length <= 0:
        return ValidationError("Square side length must be positive")
    return SquareSideLengthValidated(square_side_length)
```

Bad example:
```python
def validate_square_side_length(square_side_length: SquareSideLength) -> SquareSideLength:
    if square_side_length <= 0:
        return -1
    return square_side_length
```

8. Don't use plain int, str, bool, etc unless communicating with external boundary (libraries, code outside of the domain)
