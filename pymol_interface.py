"""
PyMOL Interface Layer

Wraps pymol2 operations into clean Python functions so the agent layer
stays free of PyMOL-specific logic.
"""

import io
import os
import sys
import pymol2

# ---------------------------------------------------------------------------
# Singleton session management
# ---------------------------------------------------------------------------

_pymol: pymol2.PyMOL | None = None


def get_session() -> pymol2.PyMOL:
    """Return the active PyMOL session, starting one if needed."""
    global _pymol
    if _pymol is None:
        _pymol = pymol2.PyMOL()
        _pymol.start()
    return _pymol


def close_session() -> None:
    """Shut down the PyMOL session."""
    global _pymol
    if _pymol is not None:
        _pymol.stop()
        _pymol = None


# ---------------------------------------------------------------------------
# Core interface functions
# ---------------------------------------------------------------------------

def load_structure(pdb_id_or_path: str) -> str:
    """
    Load a structure into the session.

    Accepts a PDB ID (fetched from RCSB) or a local file path (.pdb, .cif, etc.).
    Returns the object name PyMOL assigned to it.
    """
    cmd = get_session().cmd
    if os.path.exists(pdb_id_or_path):
        name = os.path.splitext(os.path.basename(pdb_id_or_path))[0]
        cmd.load(pdb_id_or_path, name)
    else:
        name = pdb_id_or_path.lower()
        cmd.fetch(pdb_id_or_path, name)
    return name


def get_session_state() -> str:
    """
    Serialize the current PyMOL session into a human-readable string
    suitable for injection into the LLM context.
    """
    cmd = get_session().cmd

    objects = cmd.get_object_list()
    selections = [n for n in cmd.get_names("selections")]

    lines = []

    if objects:
        lines.append("Loaded objects:")
        for obj in objects:
            n_atoms = cmd.count_atoms(obj)
            chains = cmd.get_chains(obj)
            lines.append(f"  - {obj}: {n_atoms} atoms, chains: {', '.join(chains) or 'none'}")
    else:
        lines.append("No objects loaded.")

    if selections:
        lines.append("Active selections:")
        for sel in selections:
            n_atoms = cmd.count_atoms(sel)
            lines.append(f"  - {sel}: {n_atoms} atoms")

    return "\n".join(lines)


def execute_command(cmd_string: str) -> str:
    """
    Execute an arbitrary PyMOL command string and return any captured output.

    Text that PyMOL prints to stdout (e.g. RMSD values) is captured and returned.
    """
    cmd_string = cmd_string.strip()
    if not cmd_string:
        return ""

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        get_session().cmd.do(cmd_string)
    finally:
        sys.stdout = old_stdout

    return buf.getvalue().strip()


def capture_output(pymol_cmd_string: str) -> str:
    """
    Run a PyMOL command and return its text output.

    Alias for execute_command; useful when the caller cares specifically
    about the return value (e.g. rms_cur, get_distance).
    """
    return execute_command(pymol_cmd_string)


def render_image(
    filename: str,
    width: int = 1200,
    height: int = 900,
    ray: bool = True,
    dpi: int = 300,
) -> str:
    """
    Render and save a PNG image of the current view.

    Returns the absolute path of the saved file.
    """
    cmd = get_session().cmd
    cmd.png(filename, width=width, height=height, ray=int(ray), dpi=dpi)
    return os.path.abspath(filename)


# ---------------------------------------------------------------------------
# Publication figure presets (Phase 3)
# ---------------------------------------------------------------------------

def preset_journal_standard() -> None:
    """White background, ray tracing on, antialias, suitable for most journals."""
    cmd = get_session().cmd
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 1)
    cmd.set("depth_cue", 0)
    cmd.set("ray_trace_fog", 0)


def preset_presentation() -> None:
    """Black background, ambient lighting, wider line widths for slides."""
    cmd = get_session().cmd
    cmd.bg_color("black")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("depth_cue", 1)
    cmd.set("line_width", 3)
    cmd.set("stick_radius", 0.25)


def preset_colorblind_safe() -> None:
    """
    Apply a colorblind-accessible color scheme (Wong palette) to all chains.

    Colors cycle through: blue, vermillion, bluish-green, yellow, sky-blue,
    orange, reddish-purple.
    """
    cmd = get_session().cmd
    wong_palette = [
        "0x0072B2",  # blue
        "0xD55E00",  # vermillion
        "0x009E73",  # bluish-green
        "0xF0E442",  # yellow
        "0x56B4E9",  # sky-blue
        "0xE69F00",  # orange
        "0xCC79A7",  # reddish-purple
    ]
    objects = cmd.get_object_list()
    for obj in objects:
        chains = cmd.get_chains(obj)
        for i, chain in enumerate(chains):
            color_hex = wong_palette[i % len(wong_palette)]
            cmd.color(color_hex, f"{obj} and chain {chain}")


PRESETS = {
    "journal_standard": preset_journal_standard,
    "presentation": preset_presentation,
    "colorblind_safe": preset_colorblind_safe,
}


def apply_preset(name: str) -> None:
    """Apply a named publication preset. Raises KeyError for unknown names."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    PRESETS[name]()


# ---------------------------------------------------------------------------
# Per-residue RMSD visualization (Phase 3)
# ---------------------------------------------------------------------------

def per_residue_rmsd(mobile: str, target: str) -> dict[int, float]:
    """
    Calculate per-residue RMSD between two aligned objects.

    Both objects must already be aligned (run align/super first).
    Returns a dict of {residue_number: rmsd_value}.
    Colors `mobile` by RMSD using B-factor channel (blue=low, red=high).
    """
    cmd = get_session().cmd

    # Collect CA atoms from mobile
    space = {"residues": {}}
    cmd.iterate(
        f"{mobile} and name CA",
        "residues[resi] = (model, chain, resi, resn)",
        space=space,
    )

    rmsd_by_resi: dict[int, float] = {}

    for resi, (model, chain, resi_str, resn) in space["residues"].items():
        mobile_sel = f"{mobile} and chain {chain} and resi {resi_str} and name CA"
        target_sel = f"{target} and chain {chain} and resi {resi_str} and name CA"

        # rms_cur returns RMSD as float; returns -1 if selection empty
        try:
            val = cmd.rms_cur(mobile_sel, target_sel, matchmaker=4)
        except Exception:
            val = 0.0

        if val >= 0:
            rmsd_by_resi[int(resi_str)] = val

    if not rmsd_by_resi:
        return rmsd_by_resi

    # Store RMSD values in B-factor column for coloring
    max_rmsd = max(rmsd_by_resi.values())

    def _set_bfactor(resi_str, val, mobile=mobile):
        cmd.alter(
            f"{mobile} and resi {resi_str}",
            f"b = {val}",
        )

    for resi_str, val in rmsd_by_resi.items():
        _set_bfactor(str(resi_str), val)

    cmd.spectrum("b", "blue_white_red", mobile, minimum=0, maximum=max_rmsd)

    return rmsd_by_resi


def plot_per_residue_rmsd(rmsd_by_resi: dict[int, float], output_path: str = "rmsd_plot.png") -> str:
    """
    Save a bar chart of per-residue RMSD values using matplotlib.
    Returns the absolute path of the saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    residues = sorted(rmsd_by_resi.keys())
    values = [rmsd_by_resi[r] for r in residues]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(residues, values, color="steelblue", width=1.0)
    ax.set_xlabel("Residue number")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Per-residue RMSD")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return os.path.abspath(output_path)
