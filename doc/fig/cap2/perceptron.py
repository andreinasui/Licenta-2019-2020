import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


# Instantiate the PGM.
pgm = daft.PGM()

# Hierarchical parameters.
pgm.add_node("x3", r"$x_3$", 0.5, 1)
pgm.add_node("x2", r"$x_2$", 0.5, 1.75)
pgm.add_node("x1", r"$x_1$", 0.5, 2.5)
pgm.add_node("out", r"", 2, 1.75)
pgm.add_node("iesire",r"$y$",3.25,1.75, offset=[-7,1], plot_params={"ec": "none"})

pgm.add_edge("x1","out",label=r"$w_1$")
pgm.add_edge("x2","out",label=r"$w_2$")
pgm.add_edge("x3","out",label=r"$w_3$")
pgm.add_edge("out","iesire")

# Render and save.
pgm.render()
pgm.savefig("perceptron-mine.png", dpi=300)