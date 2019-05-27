# Lesson 1 - Recap on ML

## Differentaition in n-dimensions
- In n-dim, gradient becomes jacobi matrix 
of all partial derivatives
- Linear Mapping (Linear function) (works similar to 
normal functions in coding)
    - Df(x)[v] = f(v)
        - therefore, if f(x) == x.A,
            - Df(x)[v] == v.A (or v *transpose* A)
    - Example:
        - ```
            Dict D -> {V: v, ...}
            f(x) = A.x
            Df(x)[V] = V.x
            ```

## Backprop
- This is just chain rule
- Chain rule definition:
    - derivative of outer function 
    (value of inner function)

## Bilinear mappings
- 