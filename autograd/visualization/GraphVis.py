import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
from zensor import Zensor
from uuid import uuid4

def show(G: nx.DiGraph):

    scale = 1000

    pos = graphviz_layout(G, prog='dot')
    new_pos = nx.rescale_layout(pos=np.array(list(pos.values())), scale=scale).tolist()
    for idx,key in enumerate(pos.keys()):
        pos[key] = new_pos[idx]

    nx.set_edge_attributes(G, 'color', 'r')
    labels = nx.get_node_attributes(G, 'fancy_label')

    nx.draw_networkx(G, pos=pos, arrows=True, labels=labels, edge_color='r' )

    # plt.savefig(f"plot_{uuid.uuid4()}.png", dpi=1000)

    plt.show()

def build_computation_graph(z: Zensor):
    zensor_to_instance = {}
    instance_to_zensor = {}
    back_traced = {} # zensor id / true or false

    g = nx.DiGraph()

    def create_instance(x):
        new_instance = uuid4()
        if zensor_to_instance.get(x.id):
            zensor_to_instance[x.id].append(new_instance)
        else:
            zensor_to_instance[x.id] = [new_instance]

        instance_to_zensor[new_instance] = x.id
        return new_instance

    def iter_graph(x, x_inst):
        if x.graph:
            op = x.graph.op
            for y in x.graph.ins:
                instance = create_instance(y)
                g.add_node(instance, fancy_label=f'{op}' )
                g.add_edge(instance, x_inst)

                if back_traced.get(y.id):
                    continue

                back_traced[y.id] = True
                iter_graph(y, instance)



    root_inst = create_instance(z)
    g.add_node(root_inst, fancy_label="ROOT")
    iter_graph(z, root_inst)

    return g