import lib.argParser as ArgParser
import lib.config as Config
import lib.graph as Graph
import lib.constants as Constants

# python3 main.py 
# -t Facebook 
# -a ENM 
# -f newgraph_Facebook_equal_close_ENM.txt
# -o originalgraph_Facebook.txt

def main(args):
    config = Config.Config(args.graph_type, args.attack_type)
    graph = Graph.Graph(
        config,
        prior_filename="train_0.txt",
        post_filename="post_sybilscar_0.txt",
        withBuffer=True
    )

    # initializes..
    # self.graph_file_path and set self.graph
    graph.read_graph_from_file_path(args.graph_file)
    graph.init(is_train=True)

    graph.prior_filename = "train_0.txt"
    graph.post_filename = "post_sybilscar_0_evaluation.txt"
    graph.withBuffer = False
    graph.init(is_train=True)



    og_config = Config.Config(args.graph_type, args.attack_type)
    og_graph = Graph.Graph(
        og_config,
        prior_filename="train_0.txt",
        post_filename=f"post_sybilscar_before_attack({Constants.weight}_{Constants.iteration}).txt",
        withBuffer=False
    )
    og_graph.read_graph_from_file_path(args.original_graph_file)
    og_graph.init(is_train=True)



    graph.choose_random_train_set()
    graph.trainset2prior()
    turn = 1
    graph.prior_filename = f"prior_0_prime.txt"
    graph.post_filename = f"post_sybilscar_1_prime.txt"
    graph.withBuffer = True
    graph.init(is_train=False)
    graph.check_FN_nodes()
    




if __name__ == "__main__":
    parser = ArgParser.ArgParser()
    args = parser.parse_args()
    main(args)

    