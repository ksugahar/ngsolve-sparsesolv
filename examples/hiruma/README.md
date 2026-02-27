# Hiruma Mesh Examples

Mesh files provided by Hiruma-sensei (.msh1 format).

## Mesh files

| File | Size | Description |
|------|------|-------------|
| `mesh1_2.5T.msh1` | 6.2 MB | 2.5T elements |
| `mesh1_3.5T.msh1` | 7.9 MB | 3.5T elements |
| `mesh1_4.5T.msh1` | 9.7 MB | 4.5T elements |
| `mesh1_5.5T.msh1` | 14 MB | 5.5T elements |
| `mesh1_20.5T.msh1` | 63 MB | 20.5T elements |
| `mesh1_21.5T_HF.msh1` | 165 MB | 21.5T elements (HF) |

## Region and material definitions

```json
{
    "reg_phys": {
        "core": 1,
        "cond": 2,
        "air": 3,
        "dirichlet": 4,
        "gamma_in": 5,
        "gamma_out": 6,
        "neumann": 7,
        "mix": 8
    },
    "property": {
        "core": {
            "reg_phys": 1,
            "mur": 1000,
            "sigma": 0
        },
        "cond": {
            "reg_phys": 2,
            "mur": 1,
            "sigma": 5.96e7
        },
        "air": {
            "reg_phys": 3,
            "mur": 1,
            "sigma": 0
        }
    }
}
```
