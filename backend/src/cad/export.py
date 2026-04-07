"""CAD file export (STEP/IGES)."""


def export_step(params, progress_callback=None, session=None):
    """Export the reconstructed B-Rep to STEP format."""
    output_path = params.get("output_path")
    if not output_path:
        raise ValueError("output_path is required")

    shape = session.get("brep_shape")
    if shape is None:
        raise ValueError("Must reconstruct B-Rep before exporting")

    try:
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.Interface import Interface_Static
    except ImportError:
        raise ImportError("pythonocc-core is required for STEP export")

    if progress_callback:
        progress_callback("export", 30, "Writing STEP file...")

    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP214")
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(output_path)

    if progress_callback:
        progress_callback("export", 100, "Export complete")

    return {
        "status": "ok" if status == 1 else "error",
        "output_path": output_path,
    }
