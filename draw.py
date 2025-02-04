from graphviz import Digraph
def draw(root):
    dot = Digraph(
        format="svg",
        graph_attr={
            "rankdir": "LR"
        }
    )
    def trace(root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v.prev:
                    edges.add(
                        (
                            child, 
                            v
                        )
                    )
                    build(child)
        build(root)
        return nodes, edges
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(
            id(n)
        )
        dot.node(
            name = uid, 
            label = "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), 
            shape="record"
        )
        if n.op:
            dot.node(
                name = uid + n.op, 
                label = n.op
            )
            dot.edge(
                uid + n.op, 
                uid
            )
    for n1, n2 in edges:
        dot.edge(
            str(
                id(n1)
            ), 
            str(
                id(n2)
            ) + n2.op
        )
    dot.render(
        filename = "out"
    )
