import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


# Instantiate the PGM.
pgm = daft.PGM(node_unit=0.3)

offset=-0.75

for i in range(15):
	for j in range(15):
		pgm.add_node(str(i*16+j),"",x=abs(j+(offset*j)), 
			y=abs(i+(offset*i)))


# Hierarchical parameters.
# pgm.add_node("b",r"$b$", 1.75, 2.5, fixed=True)
# pgm.add_node("x3", r"$x_3$", 0.5, 1, fixed=True)
# pgm.add_node("x2", r"$x_2$", 0.5, 1.75, fixed=True)
# pgm.add_node("x1", r"$x_1$", 0.5, 2.5, fixed=True)
# pgm.add_node("sum", r"$\sum$", 1.75, 1.75)
# pgm.add_node("fun", r"$\sigma$", 2.75, 1.75)
# pgm.add_node("iesire",r"$y$",3.75,1.75, offset=[-7,1], plot_params={"ec": "none"})

# pgm.add_edge("x1","sum",label=r"$w_1$")
# pgm.add_edge("x2","sum",label=r"$w_2$")
# pgm.add_edge("x3","sum",label=r"$w_3$")
# pgm.add_edge("b","sum")
# pgm.add_edge("sum","fun")
# pgm.add_edge("fun","iesire")

# Render and save.
pgm.render()
pgm.savefig("receptive-field.png", dpi=300)