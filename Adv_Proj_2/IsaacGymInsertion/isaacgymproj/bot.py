import PyKDL
import kdl_parser_py.urdf
class Bot:
    def __init__(self, urdfpath):
        self.urdf_path = urdfpath
        _, tree = kdl_parser_py.urdf.treeFromFile(urdfpath)
        self.chain = tree.getChain('iiwa7_link_0', 'iiwa7_link_7')
