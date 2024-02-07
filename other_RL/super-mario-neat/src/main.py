import argparse
import train as t
import run as r
import random

parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('mode', metavar='mode', type=str,
                    help="Specify 'train' or 'run' to run or train the model. To continue training, specify 'cont_train")
parser.add_argument('--gen', metavar='generations', type=int, help='Number of Generations to run for', nargs='?')
parser.add_argument('--file', metavar='file_name', type=str, help='File name to continue training or to run the winner',
                    nargs='?')
parser.add_argument('--config', metavar='config', type=str, help='Configuration File', default='config', nargs='?')
parser.add_argument('--level', metavar='level', type=str, help='Deprecated. Use --task instead.', default=None, nargs='?')
parser.add_argument('--seed', metavar='seed', type=str, help='The seed used to run the NEAT algo.', default=2, nargs='?')
parser.add_argument('--method', metavar='method', type=str, help='The method to use. Only constant, nokill, bestasref and tgraceexp are allowed.', default=None, nargs='?')
parser.add_argument('--gracetime', metavar='gracetime', type=int, help='Parameter for bestasref.', default=None, nargs='?')
parser.add_argument('--experiment_index_for_log', metavar='experiment_index_for_log', type=int, help='Parameter for log.', default=None, nargs='?')
parser.add_argument('--resultfilename', metavar='resultfilename', type=str, help='The file in which to write the results', default="resultSuperMario.txt", nargs='?')
parser.add_argument('--fincrementsize', metavar='fincrementsize', type=int, help='Fitness funcion can only increase in increments of fincrementsize.', default=None, nargs='?')
parser.add_argument('--task', metavar='task', type=str, help='Which level to run, Eg. 1-1.', nargs='?')
parser.add_argument('--max_optimization_time', metavar='max_optimization_time', type=float, help='Max runtime for experiment', default=None, nargs='?')


args = parser.parse_args()

print(args)

if (args.mode.upper() == "TRAIN" or args.mode.upper() == "CONT_TRAIN") and args.gen is None:
    parser.error("Please specify number of generations!")


if args.mode.upper() == "TRAIN":
    t = t.Train(args.method, args.gen, args.seed, args.resultfilename, args.task, args.gracetime, args.fincrementsize, experiment_index_for_log=args.experiment_index_for_log, max_optimization_time=args.max_optimization_time)
    t.main(config_file=args.config)

elif args.mode.upper() == "RUN":
    args.file = "finisher.pkl" if args.file is None else args.file
    r.main(args.config, args.file, args.task)

else:
    print("Please enter 'train' or 'mode' or 'cont_train")
