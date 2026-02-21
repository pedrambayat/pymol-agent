import pymol2

with pymol2.PyMOL() as pymol:
    pymol.cmd.fetch("1ubq")
    pymol.cmd.show("cartoon")
    pymol.cmd.png("output.png", width=1200, height=900, ray=1)