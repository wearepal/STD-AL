import itertools


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

    def add_child(self, child: "Node"):
        if child == self:
            raise ValueError("parent == child")
        child.parent = self
        self.children.append(child)
        return self

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    def __repr__(self):
        return str(self.name)

    @property
    def leafs(self):
        leafs = []
        if self.is_leaf:
            return [self]

        def _get_leaf_nodes(node: "Node"):
            if node.is_leaf:
                leafs.append(node)
            for n in node.children:
                _get_leaf_nodes(n)

        _get_leaf_nodes(self)
        return leafs

    @property
    def subnodes(self):
        nodes = []
        if self.is_leaf:
            return []

        def _get_sub_nodes(node: "Node"):
            if not node.is_leaf:
                if node != self:
                    nodes.append(node)
            for n in node.children:
                _get_sub_nodes(n)

        _get_sub_nodes(self)
        return nodes


class Tree:
    @staticmethod
    def __new__(cls, struct, n) -> Node:
        ii = itertools.count(n)
        nodes = dict()

        def create_node(name, parent=None):
            if name in nodes:
                return nodes[name]
            else:
                n = Node(name, parent)
                nodes[name] = n
                return n

        for i in struct:
            node = create_node(next(ii))
            left, right = create_node(i[0]), create_node(i[1])
            node.add_child(left).add_child(right)

        root = [n for n in nodes.values() if n.parent is None]
        assert len(root) == 1
        return root[0]
