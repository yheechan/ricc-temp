
import lib.constants as Constants

class Config:
    def __init__(
            self, graph_type, attack_type
    ):
        self.graph_type = graph_type
        self.attack_type = attack_type
        self.graph_dir_path = Constants.graph_dir_path / self.graph_type

        # initializes..
        # self.node_cnt, self.num_negative, self.num_positive,
        self.init_graph_configs()
    
    def init_graph_configs(self,):
        # what is ENM attack?
        if self.attack_type == "ENM":
            if self.graph_type == "Facebook":
                self.node_cnt = 8078
                self.num_negative = 4039
                self.num_positive = 4039
        # what is NNI attack?
        # maybe is additional of new nodes
        # New Nodes Included?
        elif self.attack_type == "NNI":
            if self.graph_type == "Facebook":
                raise Exception("Not implemented yet")
        else:
            raise Exception("Not implemented yet")

