import glob, os
import argparse
import importlib
import sys
import subprocess

from src import AutoGrid


def create_main_argparse():
    parser = argparse.ArgumentParser(description='Automatic Grid2op Experiment.')
    parser.add_argument('config_filename', metavar='configuration', help='Name of the configuration file to execute with package notation, for example ".examples.basic_example"')
    parser.add_argument("--log", dest='force_log', choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help='Force log level on all execution', default="False")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--folder", dest='execute_folder',
                       help='Treat the first parameter as a folder and execute all the configuration files inside it',
                       action='store_true')

    return parser


def check_packages():
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        pip_data = out.decode("utf-8").split("\r\n")
        pk_list = []
        for pk in pip_data:
            pk_name = pk.split("==")[0]
            pk_list.append(pk_name)

        with open("../requirements.txt", 'r', encoding='utf-8') as req_list:
            for line in req_list:
                pk_name = line.split("==")[0].replace("\n", "").replace("\r", "")
                if pk_name not in pk_list:
                    print("======================================================================================")
                    print("Package [{}] not installed, please run:".format(pk_name))
                    print("'pip install --no-index --find-links ./lib/ -r requirements.txt'")
                    print("======================================================================================")
    except:
        print("========================================")
        print("Unable to verify packages installation.")
        print("========================================")


if __name__ == "__main__":
    check_packages()
    parser = create_main_argparse()

    parsed, unknown = parser.parse_known_args()
    # this is an 'internal' method
    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg, type=str)
    args = parser.parse_args()
    if args.execute_folder == False and not importlib.util.find_spec(args.config_filename, package="experiments"):
        print(F"ERROR: Configuration file [{args.config_filename}] not found.")
        sys.exit(10)
    if args.execute_folder == True and not os.path.exists(args.config_filename):
        print(F"ERROR: Configuration folder [{args.config_filename}] not found.")
        sys.exit(20)
    force_log = args.force_log if args.force_log != "False" else False

    #Todo: pensar si la ejecucion de carpetas merece la pena, y/o si habria que paralelizarla.
    if args.execute_folder == True:
        for file in glob.glob(args.config_filename + "/*.py"):
            print("=")
            print("\r\rExecuting configuration: " + file + "\r")
            print("=")
            config_json = importlib.import_module(args.config_filename,package="experiments")
            main = AutoGrid.main(config_json, force_log=force_log)
            main.run()
    else:
        config_json = importlib.import_module(args.config_filename,package="experiments")
        main = AutoGrid.main(config_json.get_config(), force_log=force_log,
                             command_line_arguments=args)
        main.run()
