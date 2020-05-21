import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


# Instantiate the PGM.
pgm = daft.PGM()

# Hierarchical parameters.
pgm.add_node("x3", r"$x_3$", 0.5, 1.75)
pgm.add_node("x2", r"$x_2$", 0.5, 2.5)
pgm.add_node("x1", r"$x_1$", 0.5, 3.25)
pgm.add_node("input_l",r"Input Layer", 0.5, 4.75, plot_params={"ec": "none"})
pgm.add_node("x5_2", r"", 2, 1)
pgm.add_node("x4_2", r"", 2, 1.75)
pgm.add_node("x3_2", r"", 2, 2.5)
pgm.add_node("x2_2", r"", 2, 3.25)
pgm.add_node("x1_2", r"", 2, 4)
pgm.add_node("x5_3", r"", 4, 1)
pgm.add_node("x4_3", r"", 4, 1.75)
pgm.add_node("x3_3", r"", 4, 2.5)
pgm.add_node("x2_3", r"", 4, 3.25)
pgm.add_node("x1_3", r"", 4, 4)
pgm.add_node("hidden_l",r"Hidden Layers", 3, 4.75, plot_params={"ec": "none"})
pgm.add_node("out", r"", 5.5, 2.5)
pgm.add_node("output_l",r"Output Layer", 5.5, 4.75, plot_params={"ec": "none"})

pgm.add_edge("x1","x5_2")
pgm.add_edge("x1","x4_2")
pgm.add_edge("x1","x3_2")
pgm.add_edge("x1","x2_2")
pgm.add_edge("x1","x1_2")

pgm.add_edge("x2","x5_2")
pgm.add_edge("x2","x4_2")
pgm.add_edge("x2","x3_2")
pgm.add_edge("x2","x2_2")
pgm.add_edge("x2","x1_2")

pgm.add_edge("x3","x5_2")
pgm.add_edge("x3","x4_2")
pgm.add_edge("x3","x3_2")
pgm.add_edge("x3","x2_2")
pgm.add_edge("x3","x1_2")

for i in range(5):
	for j in range(5):
		pgm.add_edge(f'x{i+1}_2',f'x{j+1}_3')

pgm.add_edge("x5_3","out")
pgm.add_edge("x4_3","out")
pgm.add_edge("x3_3","out")
pgm.add_edge("x2_3","out")
pgm.add_edge("x1_3","out")

# Render and save.
pgm.render()
pgm.savefig("mlp.png", dpi=300)
