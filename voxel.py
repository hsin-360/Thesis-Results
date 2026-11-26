from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import trimesh


@dataclass
class MultiClump:
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    radii: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    diameter: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    atomid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    atomtype: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    moleid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    density: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    numSpheres: int = 0


def voxelize_single_stl(
    stl_path: Path,
    voxel_size: float,
    mesh_density: float,
    mol_id: int,
    atom_type: int = 1,
) -> Optional[dict]:
    """Voxelize one STL component (non-overlapping spheres) and compare with original STL."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
#    voxel_size = float(np.min(extents)) / int(div)

    sphere_radius = voxel_size / 2.0

    voxel_grid = mesh.voxelized(pitch=voxel_size).fill()
    occupied_indices = np.argwhere(voxel_grid.matrix)

    if len(occupied_indices) == 0:
        print(f"[Warning] div={div} too coarse; 0 voxels for {stl_path.name}")
        return None

    positions = np.asarray(voxel_grid.indices_to_points(occupied_indices), dtype=np.float64)
    n_spheres = len(positions)
    radii = np.full(n_spheres, sphere_radius, dtype=np.float64)
    diameters = np.full(n_spheres, voxel_size, dtype=np.float64)

    # 1. Original STL Properties
    if not mesh.is_watertight:
        trimesh.repair.fix_inversion(mesh)
        V_stl = abs(float(mesh.convex_hull.volume))
    else:
        V_stl = abs(float(mesh.volume))

    M_stl = V_stl * mesh_density
    com_stl = np.asarray(mesh.center_mass, dtype=np.float64)
    I_stl = np.asarray(mesh.moment_inertia, dtype=np.float64) * mesh_density

    # 2. Non-Overlapping Sphere Properties
    v_each = (4.0 / 3.0) * np.pi * (radii ** 3)
    v_spheres = float(np.sum(v_each))

    sphere_density = mesh_density * (V_stl / v_spheres)
    densities = np.full(n_spheres, sphere_density, dtype=np.float64)
    masses = v_each * sphere_density
    M_spheres = float(np.sum(masses))

    # Centroid & Parallel-Axis Inertia Tensor
    com_spheres = np.sum(positions * masses[:, np.newaxis], axis=0) / M_spheres
    rel = positions - com_spheres
    dx, dy, dz = rel[:, 0], rel[:, 1], rel[:, 2]
    I_self = (2.0 / 5.0) * masses * (radii ** 2)

    I_xx = np.sum(I_self + masses * (dy**2 + dz**2))
    I_yy = np.sum(I_self + masses * (dx**2 + dz**2))
    I_zz = np.sum(I_self + masses * (dx**2 + dy**2))
    I_xy = -np.sum(masses * dx * dy)
    I_xz = -np.sum(masses * dx * dz)
    I_yz = -np.sum(masses * dy * dz)

    I_spheres = np.array(
        [[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]],
        dtype=np.float64,
    )

    return {
        "name": stl_path.stem,
        "mol_id": mol_id,
        "atom_type": atom_type,
        "n_spheres": n_spheres,
        "positions": positions,
        "radii": radii,
        "diameters": diameters,
        "densities": densities,
        "V_stl": V_stl,
        "v_spheres": v_spheres,
        "vol_ratio": v_spheres / V_stl,
        "rho_stl": mesh_density,
        "rho_spheres": sphere_density,
        "M_stl": M_stl,
        "M_spheres": M_spheres,
        "com_stl": com_stl,
        "com_spheres": com_spheres,
        "com_err_norm": float(np.linalg.norm(com_spheres - com_stl)),
        "I_stl": I_stl,
        "I_spheres": I_spheres,
    }


def write_lammps_sphere_data(clump: MultiClump, filepath: Path, box_padding: float = 0.05) -> None:
    mins = np.min(clump.positions - clump.radii[:, np.newaxis], axis=0) - box_padding
    maxs = np.max(clump.positions + clump.radii[:, np.newaxis], axis=0) + box_padding
    n_types = int(np.max(clump.atomtype))

    with open(filepath, "w") as f:
        f.write("LAMMPS multi-STL voxelized sphere data file\n\n")
        f.write(f"{clump.numSpheres} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"{mins[0]:.8g} {maxs[0]:.8g} xlo xhi\n")
        f.write(f"{mins[1]:.8g} {maxs[1]:.8g} ylo yhi\n")
        f.write(f"{mins[2]:.8g} {maxs[2]:.8g} zlo zhi\n\n")
        f.write("Atoms # sphere\n\n")
        for i in range(clump.numSpheres):
            f.write(
                f"{clump.atomid[i]} {clump.atomtype[i]} "
                f"{clump.diameter[i]:.8g} {clump.density[i]:.8g} "
                f"{clump.positions[i, 0]:.8g} {clump.positions[i, 1]:.8g} {clump.positions[i, 2]:.8g}\n"
                         )
        f.write("\nMolecules\n\n")
        for i in range(clump.numSpheres):
            f.write(f"{clump.atomid[i]} {clump.moleid[i]}\n")
              
def write_comparison_report(parts: list[dict], filepath: Path) -> None:
    with open(filepath, "w") as f:
        f.write("=== STL vs. Voxel Non-Overlapping Spheres Comparison ===\n\n")
        for p in parts:
            f.write(f"Component: {p['name']}.stl | mol_id = {p['mol_id']} | N_spheres = {p['n_spheres']}\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'Property':<22} | {'Original STL':<24} | {'Sphere Combination':<24}\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'Volume':<22} | {p['V_stl']:<24.6g} | {p['v_spheres']:<24.6g} (ratio: {p['vol_ratio']:.4f})\n")
            f.write(f"{'Density':<22} | {p['rho_stl']:<24.6g} | {p['rho_spheres']:<24.6g}\n")
            f.write(f"{'Mass':<22} | {p['M_stl']:<24.6g} | {p['M_spheres']:<24.6g}\n")
            f.write(f"{'Centroid X':<22} | {p['com_stl'][0]:<24.6g} | {p['com_spheres'][0]:<24.6g}\n")
            f.write(f"{'Centroid Y':<22} | {p['com_stl'][1]:<24.6g} | {p['com_spheres'][1]:<24.6g}\n")
            f.write(f"{'Centroid Z':<22} | {p['com_stl'][2]:<24.6g} | {p['com_spheres'][2]:<24.6g}\n")
            f.write(f"{'Centroid |Error|':<22} | {'0.0':<24} | {p['com_err_norm']:<24.6g}\n")
            f.write("-" * 78 + "\n")
            f.write(f"Original STL Inertia Tensor (COM):\n{np.array2string(p['I_stl'], precision=6)}\n")
            f.write(f"Sphere Clump Inertia Tensor (COM):\n{np.array2string(p['I_spheres'], precision=6)}\n\n")


def build_multistl_voxel_model(
    model_dir: Path,
    output_dir: Path,
    voxel_size: float = 0.01,
    mesh_density: float = 2700.0,
    distinct_atom_types: bool = False,
    box_padding: float = 0.05,
) -> MultiClump:
    output_dir.mkdir(parents=True, exist_ok=True)
    stl_files = sorted(model_dir.glob("*.stl"))
    if not stl_files:
        raise FileNotFoundError(f"No .stl files found in {model_dir}")

    parts = []
    for idx, stl_path in enumerate(stl_files, start=1):
        print(f"[{idx}/{len(stl_files)}] Voxelizing {stl_path.name} (voxel_size={voxel_size}) -> mol-ID={idx}")
        res = voxelize_single_stl(
            stl_path=stl_path,
            voxel_size=voxel_size,
            mesh_density=mesh_density,
            mol_id=idx,
            atom_type=idx if distinct_atom_types else 1,
        )
        if res is not None:
            parts.append(res)
            print(
                f"    N={res['n_spheres']} | V_stl={res['V_stl']:.5g}, v_sph={res['v_spheres']:.5g} | "
                f"rho_sph={res['rho_spheres']:.5g} | |dCOM|={res['com_err_norm']:.4g}"
            )

    clump = MultiClump(
        positions=np.vstack([p["positions"] for p in parts]),
        radii=np.concatenate([p["radii"] for p in parts]),
        diameter=np.concatenate([p["diameters"] for p in parts]),
        density=np.concatenate([p["densities"] for p in parts]),
        moleid=np.concatenate([np.full(p["n_spheres"], p["mol_id"], dtype=np.int64) for p in parts]),
        atomtype=np.concatenate([np.full(p["n_spheres"], p["atom_type"], dtype=np.int64) for p in parts]),
    )
    clump.numSpheres = len(clump.positions)
    clump.atomid = np.arange(1, clump.numSpheres + 1, dtype=np.int64)

    write_lammps_sphere_data(clump, output_dir / "multi_clump_voxel.data", box_padding=box_padding)
    write_comparison_report(parts, output_dir / "voxel_comparison_report_size.txt")
    print(f"\n[Done] Exported LAMMPS data and comparison report to {output_dir}")
    return clump


if __name__ == "__main__":
    build_multistl_voxel_model(
        model_dir=Path("/home/joy/WORK/01_Specimen/model"),
        output_dir=Path("/home/joy/WORK/01_Specimen/lattice"),
        voxel_size = 0.1,
        mesh_density=2700.0,
    )
