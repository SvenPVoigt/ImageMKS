import sys
import os
from imagemks.workflows import cellgetparams, cellsegment, cellmeasure

def getoption(args, option):
    ind = [i for i, val in enumerate(args) if val == option]

    if len(ind) == 1:
        return args[ind[0]+1]
    else:
        raise ValueError('Multiple or none of required option (%s) found in command.'%option)


def cellanalysis_routine(args):
    subcommand = args[2]

    subcommands = [
        'info',
        'getparams',
        'segment',
        'measure',
    ]

    if subcommand == 'info':
        print("Available commands for program: " + ', '.join(subcommands))
    elif subcommand == 'getparams':
        save_p = getoption(args, '-sp')
        cellgetparams(save_p)
        print('Saved parameters to `%s`.'%save_p)
    elif subcommand == 'segment':
        path_n = getoption(args, '-fn')
        path_c = getoption(args, '-fc')
        save_n = getoption(args, '-sn')
        save_c = getoption(args, '-sc')
        path_p = getoption(args, '-fp')
        pxsize = float(getoption(args, '-z'))
        cellsegment(path_n, path_c, save_n, save_c, path_p, pxsize)
    elif subcommand == 'measure':
        path_n = getoption(args, '-fn')
        path_c = getoption(args, '-fc')
        save_m = getoption(args, '-sm')
        path_p = getoption(args, '-fp')
        pxsize = float(getoption(args, '-z'))
        cellmeasure(path_n, path_c, save_m, path_p, pxsize)
    else:
        print("Command not found. Run `cellanalysis info` for available commands.")



if __name__ == '__main__':

    commands = [
        'download',
        'info',
        'cellanalysis'
    ]

    if len(sys.argv) == 1:
        print("Available commands: " + ', '.join(commands))

    else:
        command = sys.argv[1]

        if command == 'info':
            print("Available commands for program: " + ', '.join(commands))

        elif command == 'cellanalysis':
            cellanalysis_routine(sys.argv)

        else:
            print("Command not found. Run `imagemks info` for available commands.")
