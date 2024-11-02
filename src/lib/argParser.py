import argparse

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(
            description="main.py is to run GA on SybilSCAR"
        )
        self.add_argument("-t", "--graph-type", help="graph type (e.g., Facebook, etc.)", required=True)
        self.add_argument("-a", "--attack-type", help="attack type (e.g., ENM, NNI)", required=True)
        self.add_argument("-f", "--graph-file", help="graph file path", required=True)
        self.add_argument("-o", "--original-graph-file", help="original graph file path", required=True)