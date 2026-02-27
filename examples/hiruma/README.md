# Hiruma Mesh Examples

Mesh files provided by Hiruma-sensei, converted from Gmsh v1 to v2 format (.msh).

## Mesh files

| File | Nodes | Elements | Description |
|------|-------|----------|-------------|
| `mesh1_2.5T.msh` | 23,731 | 130,460 | 2.5T elements |
| `mesh1_3.5T.msh` | 29,901 | 168,792 | 3.5T elements |
| `mesh1_4.5T.msh` | 36,473 | 205,911 | 4.5T elements |
| `mesh1_5.5T.msh` | 49,643 | 283,628 | 5.5T elements |
| `mesh1_20.5T.msh` | 211,337 | 1,232,291 | 20.5T elements |
| `mesh1_21.5T_HF.msh` | 691,541 | 2,605,376 | 21.5T elements (HF) |

Original Gmsh v1 files are in `original_msh1/` (not tracked by git).

## Usage

```python
from netgen.read_gmsh import ReadGmsh
import ngsolve

m = ReadGmsh("examples/hiruma/mesh1_2.5T")
mesh = ngsolve.Mesh(m)
```

## Region and material definitions

| Region | Physical tag | Type | mur | sigma |
|--------|-------------|------|-----|-------|
| core | 1 | Volume | 1000 | 0 |
| cond | 2 | Volume | 1 | 5.96e7 |
| air | 3 | Volume | 1 | 0 |
| dirichlet | 4 | Boundary | - | - |
| gamma_in | 5 | Boundary | - | - |
| gamma_out | 6 | Boundary | - | - |
