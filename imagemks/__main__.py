import sys
from imagemks.workflows import cellgetparams, cellsegment, cellmeasure

def search4index(args, option):
    ind = [i for i, val in enumerate(args) if val == option]

    if len(ind) == 1:
        return ind[0]
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

    if subcommand == 'getparams':
        ind = search4index(args, '-sp')
        save_p = args[ind+1]
        cellgetparams(save_p)
        print('Saved parameters to `%s`.'%save_p)
    elif subcommand == 'segment':
        path_n = search4index(args, '-fn')
        path_c = search4index(args, '-fc')
        save_n = search4index(args, '-sn')
        save_c = search4index(args, '-sc')
        path_p = search4index(args, '-fp')
        zoom = search4index(args, '-z')
        cellsegment(path_n, path_c, save_n, save_c, path_p, zoom)
    elif subcommand == 'measure':
        path_n = search4index(args, '-fn')
        path_c = search4index(args, '-fc')
        save_m = search4index(args, '-sm')
        path_p = search4index(args, '-fp')
        zoom = search4index(args, '-z')
        cellmeasure(path_n, path_c, save_m, path_p, zoom)
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
